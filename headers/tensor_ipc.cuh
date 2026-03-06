#ifndef TENSOR_IPC_CUH
#define TENSOR_IPC_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <fstream>

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// IPC handle 
const char* const IPC_HANDLE_FILE = "/tmp/shared_ipc/matrix_handle.bin";

// CUDA error handler macro 
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " Line:0 " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)
    //creating matrixes
__global__ static void init_half_matrix(__half* matrix, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) matrix[idx] = __float2half(value);
}
    //converting result matrix float 32 to float 16
__global__ void convert_float_to_half(float* in, __half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        out[idx] = __float2half(in[idx]);
    }
}
// Matrix Multiplation on tensor core
__global__ void wmma_multiply_kernel(__half *a, __half *b, float *c) {
    // Indexing
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);


    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Reseting result matrix
    wmma::fill_fragment(c_frag, 0.0f);

    // Copying data from vram to tensor registers
    wmma::load_matrix_sync(a_frag, a, WMMA_K);
    wmma::load_matrix_sync(b_frag, b, WMMA_N);

    // Tensor core calculation
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Writing result to vram
    wmma::store_matrix_sync(c, c_frag, WMMA_N, wmma::mem_row_major);
}

// IPC streaming 
inline void export_ipc_handle(void* d_ptr, const char* filename) {
    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, d_ptr));

    FILE* f = fopen(filename, "wb");
    if (!f) {
        std::cerr << "IPC Handle file did not create: " << filename << std::endl;
        exit(1);
    }
    fwrite(&handle, sizeof(cudaIpcMemHandle_t), 1, f);
    fclose(f);
    std::cout << "IPC Handle file creation is sucssesfull: " << filename << std::endl;
}

    //ipc reading 
inline void* import_ipc_handle(const char* filename) {
    cudaIpcMemHandle_t handle;
    FILE* f = fopen(filename, "rb");
    if (!f) {
        std::cerr << "IPC Handle file coulndt find" << std::endl;
        return nullptr;
    }
    fread(&handle, sizeof(cudaIpcMemHandle_t), 1, f);
    fclose(f);

    void* d_ptr;
    CUDA_CHECK(cudaIpcOpenMemHandle(&d_ptr, handle, cudaIpcMemLazyEnablePeerAccess));
    std::cout << "IPC Handle succses accsesing memory" << std::endl;
    
    return d_ptr;
}

#endif // TENSOR_IPC_CUH