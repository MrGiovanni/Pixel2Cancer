import torch
from Cellular import _C


def update_cellular(state_tensor, density_state_tensor, ranges, thresholds, flag, grow_per_cell=1, max_try=-1):
    if max_try < 0:
        max_try = grow_per_cell * 3
    
    return _CellularUpdate.apply(ranges, grow_per_cell, max_try, state_tensor, density_state_tensor, thresholds, flag)


class _CellularUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                ranges,
                grow_per_cell,
                max_try,
                state_tensor,
                density_state_tensor,
                thresholds,
                flag
        ):
        Y_range, X_range, Z_range = ranges
        organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold = thresholds
        state_tensor_new = state_tensor.clone()
        _C.update_cellular(state_tensor, density_state_tensor, Y_range, X_range, Z_range, grow_per_cell, max_try, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, flag, state_tensor_new)
        return state_tensor_new

if __name__ == '__main__':
    current_state = torch.zeros((3, 5, 5), dtype=torch.int32).cuda()
    current_state[1, 2, 2] = 1
    
    current_state = update_cellular(current_state, (3, 3, 3))
    print(current_state.sum())
    current_state = update_cellular(current_state, (3, 3, 3))
    print(current_state.sum())
    current_state = update_cellular(current_state, (3, 3, 3))
    print(current_state.sum())
