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

struct podobienstwo_liniowe
{
	statyczny_wektor<zesp> blad;
	zesp x = zesp(0.0, 0.0);

	podobienstwo_liniowe(uint64_t rozmiar)
	: blad(rozmiar)
	{}
};

__host__ podobienstwo_liniowe dopasuj_liniowo(statyczny_wektor<zesp>& A, statyczny_wektor<zesp>& B);
__host__ podobienstwo_liniowe dopasuj_liniowo_norma(statyczny_wektor<zesp>& A, statyczny_wektor<zesp>& B);

struct czy_jest_czastka{
	std::vector<podobienstwo_liniowe> dla_iteracji;

	czy_jest_czastka(spacer_losowy<zesp, TMCQ>& spacer){
		laplasjan_po_przestrzeni laplasowacz(&spacer.trwale);
		rozniczka_po_czasie rozniczkowacz(&spacer.trwale);

		statyczny_wektor<zesp> laplasowany(spacer.trwale.liczba_kubelkow());
		laplasowany.cuda_malloc();

		statyczny_wektor<zesp> rozniczkowany(spacer.trwale.liczba_kubelkow());
		rozniczkowany.cuda_malloc();

		spacer.zbuduj_na_cuda();

		for(uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++){
			laplasowany = spacer.iteracje_zapamietane[i]->wartosci;
			laplasowany.cuda_zanies(laplasowacz.stream);

			rozniczkowany = spacer.iteracje_zapamietane[i]->wartosci;
			rozniczkowany.cuda_zanies(rozniczkowacz.stream);

			laplasowacz.laplasuj(&(spacer.lokalizacja_na_device->trwale), laplasowany);
			rozniczkowacz.rozniczkuj(&(spacer.lokalizacja_na_device->trwale), rozniczkowany);

			laplasowany.cuda_przynies(laplasowacz.stream);
			rozniczkowany.cuda_przynies(rozniczkowacz.stream);

			checkCudaErrors(cudaStreamSynchronize(laplasowacz.stream));
			checkCudaErrors(cudaStreamSynchronize(rozniczkowacz.stream));

			dla_iteracji.push_back(dopasuj_liniowo_norma(laplasowany, rozniczkowany));
		}
		spacer.zburz_na_cuda();
	}
};



