import torch
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from oplanes.utils.registry import registry


@registry.register_model(name="Resnet50WithFPN")
class Resnet50WithFPN(torch.nn.Module):
    def __init__(self, withPixelDecoder=False):
        super(Resnet50WithFPN, self).__init__()
        # Get a resnet50 backbone
        m = torch.nn.Sequential(
            *(
                [
                    torch.nn.Conv2d(
                        5,
                        64,
                        kernel_size=(7, 7),
                        stride=(2, 2),
                        padding=(3, 3),
                        bias=False,
                    )
                ]
                + list(resnet50().children())[1:-2]
            )
        )
        self.body = create_feature_extractor(
            m,
            return_nodes={
                "4.2.relu_2": "1",
                "5.3.relu_2": "2",
                "6.5.relu_2": "3",
                "7.2.relu_2": "4",
            },
        )
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 5, 224, 224)
        with torch.no_grad():
            out = self.body(inp)

        self.out_channels = 256

        self.withpixeldecoder = withPixelDecoder
        if withPixelDecoder:
            input_shape = {
                k: {"stride": 224 / v.shape[-1], "channels": v.shape[-3]} for k, v in out.items()
            }  # v.shape: [b, C, h, w]
            self.pixeldecoder = PixelDecoder(input_shape)
        else:
            in_channels_list = [o.shape[1] for o in out.values()]
            # Build FPN
            out_channels = 256
            self.fpn = FeaturePyramidNetwork(
                in_channels_list,
                out_channels=out_channels,
                extra_blocks=LastLevelMaxPool(),
            )

    def forward(self, x):
        x = self.body(x)
        if self.withpixeldecoder:
            x = self.pixeldecoder(x)
        else:
            x = self.fpn(x)
        return x


class PixelDecoder(torch.nn.Module):
    """
    Ref: https://github.com/facebookresearch/detectron2/blob/6886f85baee349556749680ae8c85cdba1782d8e/detectron2/modeling/backbone/fpn.py
    """

    def __init__(self, input_shape):
        super(PixelDecoder, self).__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1]["stride"])
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v["channels"] for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        conv_dim = 256
        mask_dim = 256
        use_bias = False

        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = torch.nn.BatchNorm2d(conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    # activation=torch.nn.functional.relu,
                    activation=torch.nn.ReLU(inplace=True),
                )
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = torch.nn.BatchNorm2d(conv_dim)
                output_norm = torch.nn.BatchNorm2d(conv_dim)

                lateral_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=1,
                    bias=use_bias,
                    norm=lateral_norm,
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    # activation=torch.nn.functional.relu,
                    activation=torch.nn.ReLU(inplace=True),
                )
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = torch.nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.maskformer_num_feature_levels = 3

    def forward(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + torch.nn.functional.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), multi_scale_features


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(self.norm, torch.nn.SyncBatchNorm), "SyncBatchNorm does not support empty inputs!"

        x = torch.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
