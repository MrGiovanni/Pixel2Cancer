


#pragma once

#include <torch/extension.h>
#include <tuple>

#ifdef WITH_CUDA
at::Tensor UpdateCellular(
    const at::Tensor& state_tensor_prev,
    const at::Tensor& density_state_tensor,
    const int Y_range,
    const int X_range,
    const int Z_range,
    const int grow_per_cell,
    const int max_try,
    const int organ_hu_lowerbound,
    const int organ_standard_val,
    const int outrange_standard_val,
    const int threshold,
    const bool flag,
    at::Tensor& state_tensor // (H, W, D)
);

#else
    AT_ERROR("Not compiled with GPU support");
#endif
