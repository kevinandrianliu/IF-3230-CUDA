#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <string.h>


// Radix sort kernel
__global__ void radix_sort_kernel(const int* d_number_array, int* d_digit_array, int n, int divisor, int lock){
    __shared__ int shared_d_sub_number_array[1024];

    shared_d_sub_number_array[threadIdx.y * 32 + threadIdx.x] = 0;

    int index = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;
    if (index < n){
        shared_d_sub_number_array[index % 1024] = d_number_array[index];
    }

    __shared__ int shared_d_digit_array[10];
    if (threadIdx.y == 0 && threadIdx.x < 10){
        shared_d_digit_array[threadIdx.x] = 0;
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x == 0){
        for (int i = 0; i < 1024; i++){
            if ((blockIdx.x * 1024 + i) >= n){
                break;
            }
            shared_d_digit_array[(shared_d_sub_number_array[i] / divisor) % 10]++;
        }

        for (int i = 1; i < 10; i++){
            shared_d_digit_array[i] += shared_d_digit_array[i-1];
        }

    }
    __syncthreads();

    for (int selected_block = 0; selected_block < gridDim.x; selected_block++){
        if (blockIdx.x == selected_block){
            lock = 1;
            if (threadIdx.y == 0 && threadIdx.x < 10){
                d_digit_array[threadIdx.x] += shared_d_digit_array[threadIdx.x];
            }
            lock = 0;
        } else {
            clock_t start = clock64();
            clock_t now;
            for (;;){
                now = clock64();
                clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
                if (cycles >= 1000000){
                    break;
                }
            }

            for(;;){
                if (!lock){
                    break;
                }
            }
        }

        __syncthreads();
    }

}
__host__ void write_output(char* filename, int* h_number_array, int n){
    FILE * foutput = fopen(filename, "w");
    for (int i = 0; i < n; i++){
        fprintf(foutput, "%d\n", h_number_array[i]);
    }

    fclose(foutput);
}

__host__ void rng(int* arr, int n){
    int seed = 13516118;
    srand(seed);
    for (int i = 0; i < n; i++){
        arr[i] = (int) rand();
    }
}

int main(int argc, char** argv) {
    if (argc != 3){
        printf("Usage: %s number_of_elements file_output_name\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    if (!(n)){
        fprintf(stderr,"parameter is zero or not a number.");
        exit(EXIT_FAILURE);
    }

    int* h_number_array, *h_changed_number_array, *d_number_array, *h_digit_array, *d_digit_array;
    h_number_array = (int*) malloc (n * sizeof(int));
    if (h_number_array == NULL){
        fprintf(stderr, "Cannot allocate host memory.");
        exit(EXIT_FAILURE);
    }
    h_changed_number_array = (int *) malloc (n * sizeof(int));
    if (h_changed_number_array == NULL){
        fprintf(stderr, "Cannot allocate host memory.");
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void **) &d_number_array, n * sizeof(int));
    if (d_number_array == NULL){
        fprintf(stderr, "Cannot allocate device memory.");
        exit(EXIT_FAILURE);
    }
    cudaMemset(d_number_array, 0, n * sizeof(int));
    cudaMalloc((void **) &d_digit_array, 10 * sizeof(int));
    if (d_digit_array == NULL){
        fprintf(stderr, "Cannot allocate device memory.");
        exit(EXIT_FAILURE);
    }
    cudaMemset(d_digit_array, 0, 10 * sizeof(int));
    h_digit_array = (int *) malloc (10 * sizeof(int));
    if (h_digit_array == NULL){
        fprintf(stderr, "Cannot allocate device memory.");
        exit(EXIT_FAILURE);
    }

    rng(h_number_array,n);
    cudaMemcpy(d_number_array, h_number_array, n * sizeof(int), cudaMemcpyHostToDevice);

    int max = INT_MIN;
    for (int i = 0; i < n; i++){
        if (max < h_number_array[i]){
            max = h_number_array[i];
        }
    }

    dim3 grid, block;
    block.x = 32;
    block.y = 32;
    grid.x = n / (block.x * block.y);
    if (n % (block.x * block.y) != 0)
        grid.x++;

    int lock = 0;
    for (int divisor = 1; max/divisor > 0; divisor *= 10){
        radix_sort_kernel<<<grid, block>>>(d_number_array, d_digit_array, n, divisor, lock);
        cudaDeviceSynchronize();

        cudaMemcpy(h_digit_array, d_digit_array, 10 * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = n - 1; i >= 0; i--){
            h_changed_number_array[h_digit_array[(h_number_array[i] / divisor) % 10] - 1] = h_number_array[i];

            h_digit_array[(h_number_array[i] / divisor) % 10]--;
        }

        memcpy(h_number_array, h_changed_number_array, n * sizeof(int));
        cudaMemcpy(d_number_array, h_changed_number_array, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_digit_array, 0, 10 * sizeof(int));
    }

    write_output(argv[2], h_number_array, n);

    free(h_number_array);
    free(h_changed_number_array);
    free(h_digit_array);
    cudaFree(d_number_array);
    cudaFree(d_digit_array);
    return 0;
}