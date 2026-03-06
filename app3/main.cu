#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <tensor_ipc.cuh>
#include <unistd.h>

int main(){
    const int M = 16;
    const int N = 16;
    const int K = 16;
    //creating matrix
    float* input_matrix = nullptr;
    //we need to wait until the key file is created 
    while(input_matrix == nullptr){
    input_matrix = (float*)import_ipc_handle(IPC_HANDLE_FILE);
    sleep(1);    
}


    //allocating memory
    __half *d_a_half, *d_b_half;
    float *d_c_final;
    cudaMalloc(&d_a_half, M*K * sizeof(__half));
    cudaMalloc(&d_b_half, K*N * sizeof(__half));
    cudaMalloc(&d_c_final, M * N * sizeof(float));
    convert_float_to_half<<<1, 256>>>(input_matrix,d_a_half, M*K);
    cudaIpcCloseMemHandle(input_matrix);


    //filling empty matrix
    init_half_matrix<<<1, 256>>>(d_b_half, 0.1875f, M*K);
    //calculation
    wmma_multiply_kernel<<<1, 32>>>(d_a_half, d_b_half, d_c_final);
    cudaDeviceSynchronize();
    //showing result
    float result;
    cudaMemcpy(&result, d_c_final, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "App 3 Sonucu: " << result << std::endl;

    


}