#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


// Radix sort kernel
__global__ void radix_sort_kernel(int* d_number_array, int n){
    __shared__ int shared_d_sub_number_array[1024];

    shared_d_sub_number_array[threadIdx.y * 32 + threadIdx.x] = 0;

    int index = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;
    if (index < n){
        shared_d_sub_number_array[index % 1024] = d_number_array[index];
    }
    
    if (threadIdx.y == 0 && threadIdx.x == 0){
        printf("From block %d\n",blockIdx.x);
        for (int i = 0; i < 1024; i++){
            printf("Element %d: %d\n",i,shared_d_sub_number_array[i]);
        }
    }
}

__host__ void rng(int* arr, int n){
    int seed = 13516118;
    srand(seed);
    for (int i = 0; i < n; i++){
        arr[i] = (int) rand();
    }
}

int main(int argc, char** argv) {
    if (argc != 2){
        printf("Usage: %s number_of_elements\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    if (!(n)){
        fprintf(stderr,"parameter is zero or not a number.");
        exit(EXIT_FAILURE);
    }

    int *h_number_array, *d_number_array;
    h_number_array = (int*) malloc (n * sizeof(int));
    if (h_number_array == NULL){
        fprintf(stderr, "Cannot allocate host memory.");
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void **) &d_number_array, n * sizeof(int));
    if (d_number_array == NULL){
        fprintf(stderr, "Cannot allocate device memory.");
        exit(EXIT_FAILURE);
    }

    rng(h_number_array,n);
    cudaMemcpy(d_number_array, h_number_array, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid, block;
    block.x = 32;
    block.y = 32;
    grid.x = n / (block.x * block.y);
    if (n % (block.x * block.y) != 0)
        grid.x++;

    radix_sort_kernel<<<grid, block>>>(d_number_array, n);

    free(h_number_array);
    cudaFree(d_number_array);
    return 0;
}