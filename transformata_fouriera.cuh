#pragma once

#include <array>
#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>

#include "zesp.h"
#include "rzeczy_cudowe.h"
#include "wektory.h"

__global__ void scaling_kernel(cufftDoubleComplex* data, int element_count, double scale);
__host__ void skaluj(cudaStream_t stream, cufftDoubleComplex* data, int element_count, double scale);

struct fft_3d_po_jednym {
    cufftHandle plan;
    cudaStream_t stream = NULL;
    std::array<int, 3> dims;
    zesp* cache = nullptr;

    __host__ fft_3d_po_jednym(int szerokosc, int wysokosc, int glebokosc)
        : dims({ szerokosc, wysokosc, glebokosc }) { // moze kolejnosc jest zla
        cufftPlan3d(&plan, dims[0], dims[1], dims[2], CUFFT_Z2Z);
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cufftSetStream(plan, stream);
        checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&cache), sizeof(zesp) * rozmiar()));
    }

    __host__ void fft(statyczny_wektor<zesp>& dane) {
        dane.cuda_zanies_do(stream, cache);
        cufftExecZ2Z(plan, (cufftDoubleComplex*)cache, (cufftDoubleComplex*)cache, CUFFT_FORWARD);
        skaluj(stream, (cufftDoubleComplex*)cache, rozmiar(), std::sqrt(1.0 / rozmiar()));
        dane.cuda_przynies_z(stream, cache);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }

    __host__ void fft_inv(statyczny_wektor<zesp>& dane) {
        dane.cuda_zanies_do(stream, cache);
        skaluj(stream, (cufftDoubleComplex*)cache, rozmiar(), std::sqrt(1.0 / rozmiar()));
        cufftExecZ2Z(plan, (cufftDoubleComplex*)cache, (cufftDoubleComplex*)cache, CUFFT_INVERSE);
        dane.cuda_przynies_z(stream, cache);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }

    __host__ inline int rozmiar() {
        return dims[0] * dims[1] * dims[2];
    }

    __host__ ~fft_3d_po_jednym() {
        checkCudaErrors(cudaFree(cache));
        cufftDestroy(plan);
        checkCudaErrors(cudaStreamDestroy(stream));
        //checkCudaErrors(cudaDeviceReset()); ?
    }

};