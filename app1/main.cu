#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <tensor_ipc.cuh>
#include <unistd.h>

int main(){

    //Matrix size can be 16 minimum for tensor cores
    const int M = 16;
    const int N = 16;
    const int K = 16;


    //I/O matrixes can be float 16 so __half is used also output matrix can be 32

    __half *d_a, *d_b;
    float *d_c;

    //Allocating memory
    cudaMalloc(&d_a,M*K*sizeof(__half));
    cudaMalloc(&d_b,N*K*sizeof(__half));
    cudaMalloc(&d_c,M*N*sizeof(float));

    int threadsPerBlock = 256; 
    int blocksPerGrid = (M*K + threadsPerBlock - 1) / threadsPerBlock;

    init_half_matrix<<<blocksPerGrid, threadsPerBlock>>>(d_a, 0.03125f, M*K);
    init_half_matrix<<<blocksPerGrid, threadsPerBlock>>>(d_b, 2.0f, M*K);

    // Doldurma işleminin bitmesini bekle
    //Configuring warp
    dim3 gridDim(1,1);
    dim3 blockDim(32,1);

    //executing tensor kernel
    wmma_multiply_kernel<<<gridDim, blockDim>>>(d_a,d_b,d_c);

    cudaDeviceSynchronize();

    //giving location of c 
    export_ipc_handle(d_c, IPC_HANDLE_FILE);
    //Showing the result 
    float result;

    cudaMemcpy(&result, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "result = " << result << std::endl;
    //!!!!So there is no such a thing as lets write result to vram and leave it the procsses must be active
    while(true){
        sleep(999);
    }
    return 0; 
}