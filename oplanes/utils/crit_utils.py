import torch


def dice_loss(input_probs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor.
                The predictions for each example.
                [B, C, H, W]
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
                [B, C, H, W]
    """
    # NOTE: use sigmoid here may cause issue as values outside mask will still be considered.
    # Therefore, we need to make sure only
    # inputs = input_logits.sigmoid()  # [B, C, H, W] or [B, H, W]
    inputs = input_probs.flatten(1)  # [B, C x H x W] or [B, H x W]
    targets = targets.flatten(1)  # [B, C x H x W] or [B, H x W]
    numerator = 2 * (inputs * targets).sum(1)  # [B, ]
    denominator = inputs.sum(1) + targets.sum(1)  # [B, ]
    loss = 1 - (numerator + 1) / (denominator + 1)  # [B, ]
    # loss = loss.mean()
    return loss
