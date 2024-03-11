
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <math.h>
#include <tuple>
#include <curand_kernel.h>


// check out of range
__device__ int check_valid(
    const int H,
    const int W,
    const int D,
    const int y,
    const int x,
    const int z
){
    if (y >= H || y < 0) return 0;
    if (x >= W || x < 0) return 0;
    if (z >= D || z < 0) return 0;
    return 1;
}

// interaction with normal organ tissue
__device__ int check_tissue(
    const int target_val,
    const int organ_hu_lowerbound, 
    const int interval,
    const int density_probability
){
    // 3 level of tissue area
    const int interval_1 = organ_hu_lowerbound;
    const int interval_2 = organ_hu_lowerbound + interval;
    const int interval_3 = organ_hu_lowerbound + interval * 2;

    // probability of grow in each level organ tissue
    if(target_val == interval_1){
        if (density_probability <= 1) return 1;
    }
    else if (target_val == interval_2){
        if (density_probability < 0.3) return 1;
    }
    else if (target_val == interval_3){
        if (density_probability < 0.1) return 1;
    }
    
    return 0;
}

// mass effect: interaction with vessel and organ edge
__device__ int mass_effect(
    const int curr_val,
    const int target_val,
    const int outrange_standard_val,
    const int organ_standard_val,
    const int threshold
){
    // check whether the population of current cell is higher than the boundary/vessels invasion threshold
    if (curr_val < threshold * 10/10){
        return 1;
    }
    else {
        return 0;
    }
}


__global__ void UpdateCellularKernel(
    const int* state_tensor_prev,
    int* density_state_tensor,
    const int H,
    const int W,
    const int D,
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
    int* state_tensor // (H, W, D)
){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int interval = (outrange_standard_val - organ_hu_lowerbound) / 3;
    
    // Growth and Invasion
    if (flag == false){
        for (int pid = tid; pid < H * W * D; pid += num_threads) {
            const int curr_val = state_tensor_prev[pid];
            if (curr_val == organ_standard_val || curr_val >= outrange_standard_val){
                continue;
            }
            
            // calculate current position
            curandState state;
            const int y = pid / (W * D);
            const int x = (pid % (W * D)) / D;
            const int z = pid % D;

            // proliferative cell
            if (curr_val < threshold * 5/10){
                // self-state + 1
                atomicAdd(state_tensor + (y) * (W * D) + (x) * D + (z), 1);
                continue;
            }

            //extend cell and proliferative itself
            if (curr_val < threshold){
                atomicAdd(state_tensor + (y) * (W * D) + (x) * D + (z), 1);
            }
            
            int current_select = 0;
            int y_shift, x_shift, z_shift;
            float density_probability = 0;
            int finished = 0;
            int n_try = 0;

            while (finished < grow_per_cell && n_try < max_try){
                // random select a target neighbor
                // state += n_try * 7;
                curand_init(clock64(), tid, 0, &state);
                current_select = curand_uniform(&state) * (Y_range * X_range * Z_range + 0.9999);
                
                // current select to y x z shift
                y_shift = current_select / (X_range * Z_range);
                x_shift = current_select % (X_range * Z_range) / Z_range;
                z_shift = current_select % Z_range;

                y_shift -= (int)(Y_range / 2);
                x_shift -= (int)(X_range / 2);
                z_shift -= (int)(Z_range / 2);

                n_try ++;

                // check out of range
                if (check_valid(H, W, D, y + y_shift, x + x_shift, z + z_shift) == 0){
                    continue;
                }

                // if you want to add more rule to avoid cellular grown in certain area:
                // if (...condition...) {
                // continue;   
                // }

            
                // Simulate interaction with vessel and organ boundary
                if (mass_effect(curr_val, state_tensor[(y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift)], outrange_standard_val, organ_standard_val, threshold) == 0){
                    // Get target value
                    // if the pressure is higher than the threshold, then can invade and squeeze the vessel and organ boundary
                    if (state_tensor[(y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift)] > (outrange_standard_val + 2)){
                        state_tensor[(y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift)] = organ_standard_val;
                        density_state_tensor[(y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift)] = organ_hu_lowerbound;
                    }
                    // Apply pressure to the vessel and organ boundary
                    else if (state_tensor[(y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift)] >= (outrange_standard_val)){
                        atomicAdd(state_tensor + (y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift), 1);
                        // finished ++;
                        continue;
                    }
                }


                // Simulate interaction with normal organ tissue
                curand_init(clock64(), tid, 0, &state);
                // invasion probability
                density_probability = curand_uniform(&state);
                if (check_tissue(density_state_tensor[(y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift)], organ_hu_lowerbound, interval, density_probability) == 0){
                    continue;
                }
                


                // // connot grow in the max value area
                // if (state_tensor[(y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift)] >= threshold){
                //     continue;
                // }

            

                // connot grow in the max population area
                if (state_tensor[(y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift)] >= threshold){
                    continue;
                }


                // grow to neiguhor
                atomicAdd(state_tensor + (y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift), 1);
                finished ++;
            }
        }

    }
    // Death
    else if(flag == true){
        for (int pid = tid; pid < H * W * D; pid += num_threads) {
            const int curr_val = state_tensor_prev[pid];
            if (curr_val == -1){
                // calculate current position
                curandState state;
                const int y = pid / (W * D);
                const int x = (pid % (W * D)) / D;
                const int z = pid % D;



                int current_select = 0;
                int y_shift, x_shift, z_shift;
                float density_probability = 0;
                int finished = 0;
                int n_try = 0;
                while (finished < grow_per_cell && n_try < max_try){
                    // state += n_try * 7;
                    curand_init(clock64(), tid, 0, &state);
                    current_select = curand_uniform(&state) * (Y_range * X_range * Z_range + 0.9999);
                    
                    // current select to y x z shift
                    y_shift = current_select / (X_range * Z_range);
                    x_shift = current_select % (X_range * Z_range) / Z_range;
                    z_shift = current_select % Z_range;

                    y_shift -= (int)(Y_range / 2);
                    x_shift -= (int)(X_range / 2);
                    z_shift -= (int)(Z_range / 2);

                    n_try ++;

                    if (check_valid(H, W, D, y + y_shift, x + x_shift, z + z_shift) == 0){
                        continue;
                    }

                    if (state_tensor[(y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift)] != threshold){
                        continue;
                    }
                    state_tensor[(y + y_shift) * (W * D) + (x + x_shift) * D + (z + z_shift)] = -1;
                    finished ++;
                }
            }
            else{
                continue;
            }
        }
    }
    
}

// C++ interface for the CUDA kernel
// initialize the state_tensor
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
    
){
    at::cuda::CUDAGuard device_guard(state_tensor.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // bin_points configuration
    const int H = state_tensor.size(0);
    const int W = state_tensor.size(1);
    const int D = state_tensor.size(2);
    
    const size_t blocks = 1024;
    const size_t threads = 64;
    
    // const size_t blocks = 32;
    // const size_t threads = 8;

    // Launch the cuda kernel
    UpdateCellularKernel<<<blocks, threads, 0, stream>>>(
        state_tensor_prev.contiguous().data_ptr<int>(),
        density_state_tensor.contiguous().data_ptr<int>(),
        H,
        W,
        D,
        Y_range,
        X_range,
        Z_range,
        grow_per_cell,
        max_try,
        organ_hu_lowerbound,
        organ_standard_val,
        outrange_standard_val,
        threshold,
        flag,
        state_tensor.contiguous().data_ptr<int>() // (H, W, D)
    );
    return state_tensor;
}
