#include "rozniczka.cuh"
#include "wspolne.cuh"

constexpr __device__ double sqrt05 = 0.70710678118654752440084436210484903928483593;

__global__ void rozniczkuj_po_przestrzeni_k(spacer::dane_trwale<TMCQ>* trwale,
	zesp* zrodlo, zesp* zwrot, przydzielacz_prac przydzial){
	for (uint64_t j = 0; j < przydzial.ile_prac; j++) {
		uint64_t index_pracownika = przydzial.index_pracownika(j, threadIdx.x, blockIdx.x);

		if (index_pracownika >= przydzial.ile_prac_sumarycznie) {
			//printf("Nadmierny watek: %d", threadIdx.x);
			break;
		}

		uint64_t gdzie_wysyla = (uint64_t)trwale->gdzie_wyslac[index_pracownika];
		zesp grad = zrodlo[index_pracownika] - zrodlo[gdzie_wysyla];
		grad *= sqrt05;
		zwrot[index_pracownika] = grad; //moze powinno byæ gdzie_wysyla
	}
}

__host__ void rozniczkuj_po_przestrzeni(spacer::dane_trwale<TMCQ>* trwale,
	zesp* zrodlo, zesp* zwrot, przydzielacz_prac przydzielacz, cudaStream_t stream){
		rozniczkuj_po_przestrzeni_k start_kernel(przydzielacz, 0, stream)
		(trwale, zrodlo, zwrot, przydzielacz);
	}

__global__ void rozniczkuj_po_czasie_k(spacer::dane_trwale<TMCQ>* trwale,
	zesp* zrodlo, zesp* zwrot, przydzielacz_prac przydzial) {
	for (uint64_t j = 0; j < przydzial.ile_prac; j++) {
		uint64_t index_pracownika = przydzial.index_pracownika(j, threadIdx.x, blockIdx.x);

		if (index_pracownika >= przydzial.ile_prac_sumarycznie) {
			//printf("Nadmierny watek: %d", threadIdx.x);
			break;
		}

		spacer::info_pracownika IP = trwale->znajdz_wierzcholek(index_pracownika);

		spacer::wierzcholek& wierzcholek = trwale->wierzcholki[IP.index_wierzcholka];

		TMCQ& transformer = trwale->transformaty[wierzcholek.transformer];

		transformer.transformuj_rozniczka(trwale, wierzcholek, zrodlo, zwrot, IP.index_w_wierzcholku, IP.index_wierzcholka);
	}
}

__host__ void rozniczkuj_po_czasie(spacer::dane_trwale<TMCQ>* trwale,
	zesp* zrodlo, zesp* zwrot, przydzielacz_prac przydzielacz, cudaStream_t stream) {
	rozniczkuj_po_czasie_k start_kernel(przydzielacz, 0, stream)
		(trwale, zrodlo, zwrot, przydzielacz);
}

__global__ void laplasuj_po_przestrzeni_k(spacer::dane_trwale<TMCQ>* trwale,
	zesp* zrodlo, zesp* zwrot, przydzielacz_prac przydzial) {
	for (uint64_t j = 0; j < przydzial.ile_prac; j++) {
		uint64_t index_pracownika = przydzial.index_pracownika(j, threadIdx.x, blockIdx.x);

		if (index_pracownika >= przydzial.ile_prac_sumarycznie) {
			//printf("Nadmierny watek: %d", threadIdx.x);
			break;
		}

		spacer::info_pracownika IP = trwale->znajdz_wierzcholek(index_pracownika);

		spacer::wierzcholek& wierzcholek = trwale->wierzcholki[IP.index_wierzcholka];

		zesp laplasjan = 4 * zrodlo[index_pracownika];
		for(uint8_t i = 0; i < wierzcholek.liczba_kierunkow; i++){
			uint64_t gdzie_wysyla = (uint64_t)trwale->gdzie_wyslac[wierzcholek.start_wartosci + (spacer::idW_t)i];
			spacer::wierzcholek& sasiedni = trwale->wierzcholki[trwale->znajdz_wierzcholek(gdzie_wysyla).index_wierzcholka];

			laplasjan -= zrodlo[sasiedni.start_wartosci + (spacer::idW_t)IP.index_w_wierzcholku];
		}
		zwrot[index_pracownika] = laplasjan; //moze powinno byæ gdzie_wysyla
	}
}

__host__ void laplasuj_po_przestrzeni(spacer::dane_trwale<TMCQ>* trwale,
	zesp* zrodlo, zesp* zwrot, przydzielacz_prac przydzielacz, cudaStream_t stream) {
	laplasuj_po_przestrzeni_k start_kernel(przydzielacz, 0, stream)
		(trwale, zrodlo, zwrot, przydzielacz);
}


