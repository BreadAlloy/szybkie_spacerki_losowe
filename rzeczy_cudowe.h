#pragma once

#include <cuda_runtime.h>
//#include <cuda_profiler_api.h>

//#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
//#define _CRT_SECURE_NO_WARNINGS 1

#include <helper_functions.h>
#include <helper_cuda.h>

//#define __CUDA_ARCH__

#define __HD__ __host__ __device__

#ifdef __CUDA_ARCH__
#define IFCUDA(X, Y) X
#else
#define IFCUDA(X, Y) Y
#endif

#ifdef __CUDA_ARCH__
#define IF_HD(X, Y) Y
#else
#define IF_HD(X, Y) X
#endif

#ifdef __CUDA_ARCH__
#define IF_CUDA(X) X
#else
#define IF_CUDA(X)
#endif

#ifdef __CUDA_ARCH__
#define IF_HOST(X)
#else
#define IF_HOST(X) X
#endif

#ifdef _DEBUG
	#define sprawdzCudaErrors(val) {auto chwilowe = val; if(chwilowe){__debugbreak();} check((chwilowe), #val, __FILE__, __LINE__);}
#else
	#define sprawdzCudaErrors(val) check((val), #val, __FILE__, __LINE__);
#endif

#define if_HD_printf(wiadomosc_host, wiadomosc_cuda) IF_HD(printf(wiadomosc_cuda), printf(wiadomosc_host))

#define przenies_na_cuda(cel, zrodlo) \
sprawdzCudaErrors(cudaMalloc(reinterpret_cast<void**>(&cel), zrodlo.bajt_rozmiar())); \
sprawdzCudaErrors(cudaMemcpyAsync(reinterpret_cast<void*>(&cel), reinterpret_cast<void*>(&zrodlo), zrodlo.bajt_rozmiar(), cudaMemcpyHostToDevice, stream));

template <typename T>
struct zmienna_miedzy_HD{
	T* lokalizacja_na_cudzie = nullptr;
	T* lokalizacja_na_cpu = nullptr;
/*
	(cuda&) = lokalizacja na gpu
	(host&) = lokalizacja na cpu
*/
	__host__ zmienna_miedzy_HD(){
		lokalizacja_na_cpu = (T*)malloc(sizeof(T)); // nie inicjalizuje
		sprawdzCudaErrors(cudaMalloc(reinterpret_cast<void**>(lokalizacja_na_cudzie), sizeof(T)));
	}
		
	__host__ T* adres_cuda(){
		return lokalizacja_na_cudzie;
	}

	__host__ T* adres_host(){
		return lokalizacja_na_cpu;
	}

	__host__ ~zmienna_miedzy_HD(){
		free(lokalizacja_na_cpu);
		sprawdzCudaErrors(cudaFree(lokalizacja_na_cudzie));
	}
	
};
