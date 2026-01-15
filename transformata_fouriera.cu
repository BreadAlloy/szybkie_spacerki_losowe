#include "transformata_fouriera.cuh"

//#ifndef CUDA_RT_CALL
//#define CUDA_RT_CALL( call )                                                                                           \
//    {                                                                                                                  \
//        auto status = static_cast<cudaError_t>( call );                                                                \
//        if ( status != cudaSuccess )                                                                                   \
//            fprintf( stderr,                                                                                           \
//                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
//                     "with "                                                                                           \
//                     "%s (%d).\n",                                                                                     \
//                     #call,                                                                                            \
//                     __LINE__,                                                                                         \
//                     __FILE__,                                                                                         \
//                     cudaGetErrorString( status ),                                                                     \
//                     status );                                                                                         \
//    }
//#endif  // CUDA_RT_CALL
//
//// cufft API error chekcing
//#ifndef CUFFT_CALL
//#define CUFFT_CALL( call )                                                                                             \
//    {                                                                                                                  \
//        auto status = static_cast<cufftResult>( call );                                                                \
//        if ( status != CUFFT_SUCCESS )                                                                                 \
//            fprintf( stderr,                                                                                           \
//                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
//                     "with "                                                                                           \
//                     "code (%d).\n",                                                                                   \
//                     #call,                                                                                            \
//                     __LINE__,                                                                                         \
//                     __FILE__,                                                                                         \
//                     status );                                                                                         \
//    }
//#endif  // CUFFT_CALL

__global__ void scaling_kernel(cufftDoubleComplex* data, int element_count, double scale) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (auto i = tid; i < element_count; i += stride) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

__host__ void skaluj(cudaStream_t stream, cufftDoubleComplex* data, int element_count, double scale){
    scaling_kernel<<<1, 128, 0, stream>>>((cufftDoubleComplex*)data, element_count, scale);
}

#if 0

using dim_t = std::array<int, 3>;

int main() {
    cufftHandle plan;
    cudaStream_t stream = NULL;

    int n = 7;
    dim_t fft = { 500, 500, 3 };
    int batch_size = 1;
    int fft_size = fft[0] * fft[1] * fft[2];

    using scalar_type = double;
    using data_type = std::complex<scalar_type>;

    std::vector<data_type> data(fft_size * batch_size);

    for (int i = 0; i < data.size(); i++) {
        data[i] = data_type(i, -i);
    }

    std::printf("Input array:\n");
    for (int i = 0; i < 10; i++) {
        std::printf("%f + %fj\n", data[i].real(), data[i].imag());
    }
    std::printf("=====\n");

    cufftDoubleComplex* d_data = nullptr;

    // inembed/onembed being nullptr indicates contiguous data for each batch, then the stride and dist settings are ignored
    cufftPlan3d(&plan, fft[0], fft[1], fft[2], CUFFT_Z2Z);

    sprawdzCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    sprawdzCudaErrors(cufftSetStream(plan, stream));

    // Create device data arrays
    sprawdzCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(data_type) * data.size()));
    sprawdzCudaErrors(cudaMemcpyAsync(d_data, data.data(), sizeof(data_type) * data.size(), cudaMemcpyHostToDevice, stream));

    /*
     * Note:
     *  Identical pointers to data and output arrays implies in-place transformation
     */
    sprawdzCudaErrors(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));
    scaling_kernel << <1, 128, 0, stream >> > (d_data, data.size(), 1.0 / fft_size);
    sprawdzCudaErrors(cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(), cudaMemcpyDeviceToHost, stream));
    sprawdzCudaErrors(cudaStreamSynchronize(stream));

    std::printf("Output array after Forward transform:\n");
    for (int i = 0; i < 10; i++) {
        std::printf("%f + %fj\n", data[i].real(), data[i].imag());
    }
    std::printf("=====\n");

    // Normalize the data and inverse FFT
    sprawdzCudaErrors(cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE));
    sprawdzCudaErrors(cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(),
        cudaMemcpyDeviceToHost, stream));

    sprawdzCudaErrors(cudaStreamSynchronize(stream));

    std::printf("Output array after Forward, Normalization and Inverse transform:\n");
    for (int i = 0; i < 10; i++) {
        std::printf("%f + %fj\n", data[i].real(), data[i].imag());
    }
    std::printf("=====\n");



    /* free resources */
    sprawdzCudaErrors(cudaFree(d_data));
    sprawdzCudaErrors(cufftDestroy(plan));
    sprawdzCudaErrors(cudaStreamDestroy(stream));
    sprawdzCudaErrors(cudaDeviceReset());

    return EXIT_SUCCESS;
}

#endif




