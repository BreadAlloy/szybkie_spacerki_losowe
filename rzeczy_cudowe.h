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

struct przydzielacz_prac {
	uint64_t ile_prac = 0;
	uint64_t ile_watkow = 0;
	uint64_t ile_blokow = 0;
	uint64_t ile_prac_sumarycznie = 0;

	__host__ przydzielacz_prac(uint64_t ile_prac_sumarycznie, uint64_t ile_prac_na_watek, uint64_t max_ile_watkow) :
		ile_prac_sumarycznie(ile_prac_sumarycznie), ile_prac(ile_prac_na_watek) {
		uint64_t ile_watkow_sumarycznie = ile_prac_sumarycznie / ile_prac_na_watek + 1;
		ile_blokow = ile_watkow_sumarycznie / max_ile_watkow + 1;
		ile_watkow = ile_watkow_sumarycznie / ile_blokow + 1;
	}

	__device__ __forceinline__ uint64_t index_pracownika(uint64_t index_pracy, uint64_t index_watka, uint64_t index_bloku) {
		return ile_watkow * (ile_prac * index_bloku + index_pracy) + index_watka;
		//return ile_prac * (ile_watkow * index_bloku + index_watka) + index_pracy; // o wiele gorsze
	}
};

#define start_kernel(przydzielacz, rozmiar_pamieci_dzielonej, stream) <<<(uint32_t)przydzielacz.ile_blokow, (uint32_t)przydzielacz.ile_watkow, rozmiar_pamieci_dzielonej, stream>>>

//template <typename ptr_type>
//struct cuda_ptr{
//	ptr_type* ptr;
//
//	ptr_type operator*(){
//		IFNOTCUDA(){
//			static_assert(false);
//		}
//		return *ptr;
//	}
//};