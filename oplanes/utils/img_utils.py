import numpy as np
from typing import List

import torch


def crop_vis_map_nonsqure(vis_map):
    assert vis_map.ndim == 2, f"{vis_map.shape}"
    h, w = vis_map.shape

    nonzero_rows, nonzero_cols = torch.nonzero(vis_map, as_tuple=True)
    min_row = torch.min(nonzero_rows)
    max_row = torch.max(nonzero_rows)
    min_col = torch.min(nonzero_cols)
    max_col = torch.max(nonzero_cols)

    append_ratio = 0.05
    n_append_rows = int(append_ratio * (max_row - min_row))
    n_append_cols = int(append_ratio * (max_col - min_col))
    start_row = max(0, min_row - n_append_rows)
    end_row = min(h, max_row + n_append_rows)
    start_col = max(0, min_col - n_append_cols)
    end_col = min(w, max_col + n_append_cols)

    return start_row, end_row, start_col, end_col, min_row, max_row, min_col, max_col


def compute_prepend_and_append(min_pos, max_pos, tgt_len, total_len):
    cur_len = max_pos - min_pos
    n_half_pad = int((tgt_len - cur_len) / 2)
    if n_half_pad > min_pos:
        n_prepend = min_pos
        n_append = tgt_len - cur_len - n_prepend
    else:
        n_prepend = n_half_pad
        n_append = tgt_len - cur_len - n_prepend
        if n_append > total_len - max_pos:
            n_append = total_len - max_pos
            n_prepend = tgt_len - cur_len - n_append
    return n_prepend, n_append


def crop_vis_map(vis_map, expand_ratio=0.05):
    assert vis_map.ndim == 2, f"{vis_map.shape}"
    h, w = vis_map.shape

    nonzero_rows, nonzero_cols = torch.nonzero(vis_map, as_tuple=True)
    # NOTE: the following max_row/col are excluded
    min_row = torch.min(nonzero_rows).item()
    max_row = torch.max(nonzero_rows).item() + 1
    min_col = torch.min(nonzero_cols).item()
    max_col = torch.max(nonzero_cols).item() + 1

    # NOTE: we must make the cropped image square.
    # Otherwise, we cannot recover the original "aspect ratio" after marching cubes.
    n_rows = max_row - min_row
    n_cols = max_col - min_col

    if n_rows > w:
        n_sqaure_pixs = w
        min_row = np.random.randint(min_row, max_row - w)
        max_row = min_row + w
        n_rows = max_row - min_row
    elif n_cols > h:
        n_sqaure_pixs = h
        min_col = np.random.randint(min_col, max_col - h)
        max_col = min_col + h
        n_cols = max_col - min_col
    else:
        n_sqaure_pixs = max(n_rows, n_cols)
        n_sqaure_pixs = min(int((1 + expand_ratio) * n_sqaure_pixs), h, w)

    assert n_sqaure_pixs >= n_rows, f"{n_sqaure_pixs}, {n_rows}, {min_row}, {max_row}"
    assert n_sqaure_pixs >= n_cols, f"{n_sqaure_pixs}, {n_cols}, {min_col}, {max_col}"

    n_top_append_rows, n_bottom_append_rows = compute_prepend_and_append(min_row, max_row, n_sqaure_pixs, h)
    n_left_append_cols, n_right_append_cols = compute_prepend_and_append(min_col, max_col, n_sqaure_pixs, w)

    start_row = max(0, min_row - n_top_append_rows)
    end_row = min(h, max_row + n_bottom_append_rows)
    start_col = max(0, min_col - n_left_append_cols)
    end_col = min(w, max_col + n_right_append_cols)

    assert (end_row - start_row) == (end_col - start_col), f"{start_row}, {end_row}, {start_col}, {end_col}"
    # print("square: ", end_row - start_row)

    return start_row, end_row, start_col, end_col, min_row, max_row, min_col, max_col


# -----------------------------------------------------------------------------------
# Filters

# modified from https://github.com/kornia/kornia/blob/3606cf9c3d1eb3aabd65ca36a0e7cb98944c01ba/kornia/filters/filter.py#L32


Sobel_X = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
Sobel_Y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Computes padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) >= 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding


def filter2D(in_tensor, kernel):
    """
    in_tensor: [B, in_C, H, W]
    kernel: [B, kH, kW]
    """
    b, c, h, w = in_tensor.shape
    tmp_kernel = kernel.unsqueeze(1).to(in_tensor)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1).contiguous()
    # print("tmp_kernel: ", tmp_kernel.shape)

    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape = _compute_padding([height, width])
    input_pad = torch.nn.functional.pad(in_tensor, padding_shape, mode="reflect")
    # print("input_pad: ", input_pad.shape)

    out_tensor = torch.nn.functional.conv2d(input_pad, tmp_kernel, padding=0, stride=1)
    # print("out_tensor: ", out_tensor.shape)

    return out_tensor


def depth2normal(depth, kernel_x, kernel_y):
    """
    in_tensor: [B, in_C, H, W]
    kernel: [B, kH, kW]

    Ref: https://github.com/ZhengZerong/DeepHuman/blob/46ca0916d4531cf111c3a0d334d73d59773f0474/TrainerNormal.py#L551
    """

    kernel_x = kernel_x.to(depth.device)
    kernel_y = kernel_y.to(depth.device)

    bs, _, h, w = depth.shape

    # [H, W]
    h_grid, w_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    h_grid = h_grid.float().unsqueeze(0).unsqueeze(0).expand(bs, -1, -1, -1).to(depth.device)
    w_grid = w_grid.float().unsqueeze(0).unsqueeze(0).expand(bs, -1, -1, -1).to(depth.device)

    h_grid = (h_grid / h) * 2 - 1
    w_grid = (w_grid / w) * 2 - 1

    # X-axis corresponds to horizontal: columns
    w_grid_dx = filter2D(w_grid, kernel_x)
    h_grid_dx = filter2D(h_grid, kernel_x)
    depth_dx = filter2D(depth, kernel_x)
    # [B, 3, H, W]
    dx = torch.cat([w_grid_dx, h_grid_dx, depth_dx], axis=1)

    w_grid_dy = filter2D(w_grid, kernel_y)
    h_grid_dy = filter2D(h_grid, kernel_y)
    depth_dy = filter2D(depth, kernel_y)
    # [B, 3, H, W]
    dy = torch.cat([w_grid_dy, h_grid_dy, depth_dy], axis=1)

    normal = torch.cross(dy, dx, dim=1)
    normal = normal / (torch.norm(normal, dim=1, keepdim=True, p=2) + 1e-8)
    # # [B, 1, H, W]
    # rgb_grad = (classic_grad_x + classic_grad_y) / 2

    # # reverse the orientation
    # normal = 2 * normal - 1
    # normal = -1 * normal
    # normal = (normal + 1) / 2

    return normal
