#pragma once
#include "spacer_losowy.h"
#include "transformaty_wyspecializowane.h"

__host__ void rozniczkuj_po_przestrzeni(spacer::dane_trwale<TMCQ>* trwale,
	zesp* zrodlo, zesp* zwrot, przydzielacz_prac przydzielacz, cudaStream_t stream);
__host__ void rozniczkuj_po_czasie(spacer::dane_trwale<TMCQ>* trwale,
	zesp* zrodlo, zesp* zwrot, przydzielacz_prac przydzielacz, cudaStream_t stream);
__host__ void laplasuj_po_przestrzeni(spacer::dane_trwale<TMCQ>* trwale,
	zesp* zrodlo, zesp* zwrot, przydzielacz_prac przydzielacz, cudaStream_t stream);

struct rozniczka_po_przestrzeni{
	zesp* cache = nullptr;
	cudaStream_t stream = NULL;
	spacer::dane_trwale<TMCQ>* trwale = nullptr;
	przydzielacz_prac przydzielacz;

	__host__ rozniczka_po_przestrzeni(spacer::dane_trwale<TMCQ>* trwale)
	: trwale(trwale), przydzielacz(trwale->liczba_kubelkow(), 30, 500) {
		sprawdzCudaErrors(cudaMalloc(reinterpret_cast<void**>(&cache), sizeof(zesp) * trwale->liczba_kubelkow()));
		sprawdzCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	}

	// dane musi byc na cudzie
	__host__ void rozniczkuj(spacer::dane_trwale<TMCQ>* trwale_gpu, statyczny_wektor<zesp>& dane){
		ASSERT_Z_ERROR_MSG(trwale->liczba_kubelkow() == dane.rozmiar, "Niepoprawny rozmiar danych\n");
		ASSERT_Z_ERROR_MSG(dane.pamiec_device != nullptr, "Nie jest na cudzie\n");
		rozniczkuj_po_przestrzeni(trwale_gpu, dane.pamiec_device, cache, przydzielacz, stream);
		zesp* temp = cache;
		cache = dane.pamiec_device;
		dane.pamiec_device = temp;
	}

	__host__ ~rozniczka_po_przestrzeni(){
		sprawdzCudaErrors(cudaFree(cache));
		sprawdzCudaErrors(cudaStreamDestroy(stream));
	}
};

struct rozniczka_po_czasie {
	zesp* cache = nullptr;
	cudaStream_t stream = NULL;
	spacer::dane_trwale<TMCQ>* trwale = nullptr;
	przydzielacz_prac przydzielacz;

	__host__ rozniczka_po_czasie(spacer::dane_trwale<TMCQ>* trwale)
		: trwale(trwale), przydzielacz(trwale->liczba_kubelkow(), 30, 500) { // mo¿e powinno byæ zalezne od ilosci watkow
		sprawdzCudaErrors(cudaMalloc(reinterpret_cast<void**>(&cache), sizeof(zesp) * trwale->liczba_kubelkow()));
		sprawdzCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	}

	// dane musi byc na cudzie
	__host__ void rozniczkuj(spacer::dane_trwale<TMCQ>* trwale_gpu, statyczny_wektor<zesp>& dane) {
		ASSERT_Z_ERROR_MSG(trwale->liczba_kubelkow() == dane.rozmiar, "Niepoprawny rozmiar danych\n");
		ASSERT_Z_ERROR_MSG(dane.pamiec_device != nullptr, "Nie jest na cudzie\n");
		rozniczkuj_po_czasie(trwale_gpu, dane.pamiec_device, cache, przydzielacz, stream);
		zesp* temp = cache;
		cache = dane.pamiec_device;
		dane.pamiec_device = temp;
	}

	__host__ ~rozniczka_po_czasie() {
		sprawdzCudaErrors(cudaFree(cache));
		sprawdzCudaErrors(cudaStreamDestroy(stream));
	}
};

struct laplasjan_po_przestrzeni {
	zesp* cache = nullptr;
	cudaStream_t stream = NULL;
	spacer::dane_trwale<TMCQ>* trwale = nullptr;
	przydzielacz_prac przydzielacz;

	__host__ laplasjan_po_przestrzeni(spacer::dane_trwale<TMCQ>* trwale)
		: trwale(trwale), przydzielacz(trwale->liczba_kubelkow(), 30, 500) {
		sprawdzCudaErrors(cudaMalloc(reinterpret_cast<void**>(&cache), sizeof(zesp) * trwale->liczba_kubelkow()));
		sprawdzCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	}

	// dane musi byc na cudzie
	__host__ void laplasuj(spacer::dane_trwale<TMCQ>* trwale_gpu, statyczny_wektor<zesp>& dane) {
		ASSERT_Z_ERROR_MSG(trwale->liczba_kubelkow() == dane.rozmiar, "Niepoprawny rozmiar danych\n");
		ASSERT_Z_ERROR_MSG(dane.pamiec_device != nullptr, "Nie jest na cudzie\n");
		laplasuj_po_przestrzeni(trwale_gpu, dane.pamiec_device, cache, przydzielacz, stream);
		zesp* temp = cache;
		cache = dane.pamiec_device;
		dane.pamiec_device = temp;
	}

	__host__ ~laplasjan_po_przestrzeni() {
		sprawdzCudaErrors(cudaFree(cache));
		sprawdzCudaErrors(cudaStreamDestroy(stream));
	}
};




