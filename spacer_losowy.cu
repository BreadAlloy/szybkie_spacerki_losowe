#include "spacer_losowy.h"

#include "transformaty_wyspecializowane.h"

//#include <cooperative_groups.h> nie mam compute compatibility 9.0 :( cluster.sync()

__HD__ __forceinline__ double dot(const estetyczny_wektor<double>& a, const estetyczny_wektor<double>& b) {
	ASSERT_Z_ERROR_MSG(a.rozmiar == b.rozmiar, "Dot product na innej ilosci elementow\n");
	double sum = 0.0;
	for (uint64_t i = 0; i < a.rozmiar; i++) {
		sum += (a[i] * b[i]);
	}
	return sum;
}

__HD__ __forceinline__ zesp dot(const estetyczny_wektor<zesp>& a, const estetyczny_wektor<zesp>& b) {
	ASSERT_Z_ERROR_MSG(a.rozmiar == b.rozmiar, "Dot product na innej ilosci elementow\n");
	zesp sum = zesp(0.0, 0.0);
	for (uint64_t i = 0; i < a.rozmiar; i++) {
		sum += (a[i].sprzezenie() * b[i]);
	}
	return sum;
}

struct przydzielacz_prac{
	uint64_t ile_prac = 0;
	uint64_t ile_watkow = 0;
	uint64_t ile_blokow = 0;
	uint64_t ile_prac_sumarycznie = 0;

	przydzielacz_prac(uint64_t ile_prac_sumarycznie, uint64_t ile_prac_na_watek, uint64_t max_ile_watkow) : 
		ile_prac_sumarycznie(ile_prac_sumarycznie), ile_prac(ile_prac_na_watek){
		uint64_t ile_watkow_sumarycznie = ile_prac_sumarycznie / ile_prac_na_watek + 1;
		ile_blokow = ile_watkow_sumarycznie / max_ile_watkow + 1;
		ile_watkow = ile_watkow_sumarycznie / ile_blokow + 1;
	}

	__device__ __forceinline__ uint64_t index_pracownika(uint64_t index_pracy, uint64_t index_watka, uint64_t index_bloku){
		return ile_watkow * (ile_prac * index_bloku + index_pracy) + index_watka;
		//return ile_prac * (ile_watkow * index_bloku + index_watka) + index_pracy; // o wiele gorsze
	}
};

#define start_kernel(przydzielacz, rozmiar_pamieci_dzielonej, stream) <<<(uint32_t)przydzielacz.ile_blokow, (uint32_t)przydzielacz.ile_watkow, rozmiar_pamieci_dzielonej, stream>>>

// na wiele blokow
template<typename towar>
__global__ void suma_czesc_1(towar* sumowany, towar* cache, przydzielacz_prac przydzial) {
	extern __shared__ towar podsumy[];
	towar suma = zero(towar());
	for (uint64_t j = 0; j < przydzial.ile_prac; j++) {
		uint64_t index_pracownika = przydzial.index_pracownika(j, threadIdx.x, blockIdx.x);
		if(index_pracownika >= przydzial.ile_prac_sumarycznie){break;}

		suma += sumowany[index_pracownika];
	}
	podsumy[threadIdx.x] = suma;
	__syncthreads();
	if(threadIdx.x == 0){
		suma = zero(towar());
		for (uint64_t j = 0; j < przydzial.ile_watkow; j++) {
			suma += podsumy[j];
		}
		cache[blockIdx.x] = suma;
	}
}

// na wiele blokow
template<typename towar>
__global__ void suma_P_czesc_1(towar* sumowany, double* cache, przydzielacz_prac przydzial) {
	extern __shared__ double podsumy[];
	double suma = 0.0;
	for (uint64_t j = 0; j < przydzial.ile_prac; j++) {
		uint64_t index_pracownika = przydzial.index_pracownika(j, threadIdx.x, blockIdx.x);
		if (index_pracownika >= przydzial.ile_prac_sumarycznie){break;}

		suma += P(sumowany[index_pracownika]);
	}
	podsumy[threadIdx.x] = suma;
	__syncthreads();
	if (threadIdx.x == 0) {
		suma = 0.0;
		for (uint64_t j = 0; j < przydzial.ile_watkow; j++) {
			suma += podsumy[j];
		}
		cache[blockIdx.x] = suma;
	}
}

template<typename towar>
__global__ void suma_czesc_2(towar* cache, uint64_t rozmiar, towar* cel) {
	if(blockIdx.x == 0 && threadIdx.x == 0){
		towar suma = zero(towar());
		for(uint64_t i = 0; i < rozmiar; i++){
			suma += cache[i];
		}
		*cel = suma;
	}
}

template<typename towar>
struct gpu_sumator{
	towar* cache;
	uint64_t rozmiar_sumowanego = 0;
	przydzielacz_prac przydzielacz;

	__host__ gpu_sumator(uint64_t rozmiar_sumowanego, uint64_t max_ile_watkow = 500, uint64_t ile_prac_na_watek = 10)
	: rozmiar_sumowanego(rozmiar_sumowanego), przydzielacz(rozmiar_sumowanego, ile_prac_na_watek, max_ile_watkow){
		sprawdzCudaErrors(cudaMalloc((void**)&cache, sizeof(towar) * przydzielacz.ile_blokow));
	}

	__host__ void sumuj(const statyczny_wektor<towar>& sumowany, towar* adres_zwrotu, cudaStream_t stream){
		if(rozmiar_sumowanego != 0){
			ASSERT_Z_ERROR_MSG(rozmiar_sumowanego == sumowany.rozmiar, "Nie jest to ten sam rozmiar\n");
			suma_P_czesc_1<towar>start_kernel(przydzielacz, przydzielacz.ile_watkow * sizeof(towar), stream)
				(sumowany.pamiec_device, cache, przydzielacz);
			suma_czesc_2<towar><<<1,1,0,stream>>>
				(cache, przydzielacz.ile_blokow, adres_zwrotu);
		}
	}

	__host__ ~gpu_sumator(){
		sprawdzCudaErrors(cudaFree((void*)cache));
	}
};

template<typename towar>
struct gpu_P_sumator {
	double* cache;
	uint64_t rozmiar_sumowanego = 0;
	przydzielacz_prac przydzielacz;

	__host__ gpu_P_sumator(uint64_t rozmiar_sumowanego, uint64_t max_ile_watkow = 500, uint64_t ile_prac_na_watek = 10)
		: rozmiar_sumowanego(rozmiar_sumowanego), przydzielacz(rozmiar_sumowanego, ile_prac_na_watek, max_ile_watkow) {
		sprawdzCudaErrors(cudaMalloc((void**)&cache, sizeof(double) * przydzielacz.ile_blokow));
	}

	__host__ void sumuj(const statyczny_wektor<towar>& sumowany, double* adres_zwrotu, cudaStream_t stream) {
		if (rozmiar_sumowanego != 0) {
			ASSERT_Z_ERROR_MSG(rozmiar_sumowanego == sumowany.rozmiar, "Nie jest to ten sam rozmiar\n");
			suma_P_czesc_1<towar>start_kernel(przydzielacz, przydzielacz.ile_watkow * sizeof(double), stream)
				(sumowany.pamiec_device, cache, przydzielacz);
			suma_czesc_2<double><<<1, 1, 0, stream>>>
				(cache, przydzielacz.ile_blokow, adres_zwrotu);
		}
	}

	__host__ ~gpu_P_sumator() {
		sprawdzCudaErrors(cudaFree((void*)cache));
	}
};

template <typename towar, typename transformata>
__global__ void iteracja_na_gpu(spacer::dane_trwale<transformata>* trwale, spacer::dane_iteracji<towar>* iteracja_z, spacer::dane_iteracji<towar>* iteracja_do, przydzielacz_prac przydzial) {
	for(uint64_t j = 0; j < przydzial.ile_prac; j++){
		uint64_t index_pracownika = przydzial.index_pracownika(j, threadIdx.x, blockIdx.x);
	
		if (index_pracownika >= przydzial.ile_prac_sumarycznie) {
			//printf("Nadmierny watek: %d", threadIdx.x);
			break;
		}

		spacer::info_pracownika IP = trwale->znajdz_wierzcholek(index_pracownika);

		uint32_t index_wierzcholka = IP.index_wierzcholka;
		uint8_t index_w_wierzcholku = IP.index_w_wierzcholku;

		spacer::wierzcholek& wierzcholek = trwale->wierzcholki[index_wierzcholka];

		trwale->transformaty[wierzcholek.transformer].transformuj(*trwale, wierzcholek, *iteracja_z, *iteracja_do, index_w_wierzcholku, index_wierzcholka);
	}
}

template <typename towar, typename transformata>
__global__ void absorbuj_na_gpu(spacer::dane_trwale<transformata>* trwale, spacer::dane_iteracji<towar>* iteracja_z, spacer::dane_iteracji<towar>* iteracja_do, double procent_absorbowany,
	przydzielacz_prac przydzial){

	double norma_zabranego = NORMA(1.0, procent_absorbowany, towar()); // nie jest inline
	double norma_pozostawionego = NORMA(1.0, 1.0 - procent_absorbowany, towar()); // nie jest inline

	for (uint64_t j = 0; j < przydzial.ile_prac; j++) {
		uint64_t index_pracownika = przydzial.index_pracownika(j, threadIdx.x, blockIdx.x);

		if(index_pracownika >= przydzial.ile_prac_sumarycznie){
			break;
		}
		uint64_t indeks_absorbowany = trwale->indeksy_absorbowane[index_pracownika];
		towar zaabsorbowane = iteracja_do->wartosci[indeks_absorbowany] * norma_zabranego;
		iteracja_do->wartosci[indeks_absorbowany] *= norma_pozostawionego;
		iteracja_do->wartosci_zaabsorbowane[index_pracownika] = P(zaabsorbowane) + iteracja_z->wartosci_zaabsorbowane[index_pracownika];
	}
}

template <typename towar>
__global__ void policz_wspolczynnik_normalizacji(spacer::dane_iteracji<towar>* iteracja_z, spacer::dane_iteracji<towar>* iteracja_do, double poczatkowe_prawdopodobienstwo = 1.0) { //na jeden watek w jednym bloku
	iteracja_do->zaabsorbowane_poprzedniej += iteracja_do->zaabsorbowane_poprzedniej;
	double powinno_byc = poczatkowe_prawdopodobienstwo - iteracja_do->zaabsorbowane_poprzedniej;
	iteracja_do->norma_poprzedniej_iteracji = NORMA(iteracja_do->prawdopodobienstwo_poprzedniej, powinno_byc, towar());
}

template <typename towar, typename transformata>
__global__ void zakoncz_iteracje(spacer::dane_iteracji<towar>* iteracja_do, double t) { //na jeden watek w jednym bloku
	iteracja_do->czas = t;
}

template <typename towar>
__global__ void nie_normalizuj(spacer::dane_iteracji<towar>* iteracja_z, spacer::dane_iteracji<towar>* iteracja_do) { //na jeden watek w jednym bloku
	iteracja_do->prawdopodobienstwo_poprzedniej = iteracja_z->prawdopodobienstwo_poprzedniej;
	iteracja_do->zaabsorbowane_poprzedniej = iteracja_z->zaabsorbowane_poprzedniej;
	iteracja_do->norma_poprzedniej_iteracji = 1.0;
}

template <typename towar, typename transformata>
__host__ void iteracje_na_gpu(spacer_losowy<towar, transformata>& spacer, double delta_t,
	uint64_t liczba_iteracji, uint64_t ile_prac_na_watek, uint32_t ile_watkow_na_blok_max, uint32_t co_ile_zapisac, uint32_t co_ile_normalizuj, uint32_t co_ile_absorbuj) {

	przydzielacz_prac przydzielacz_iteracja(spacer.trwale.ile_prac(), ile_prac_na_watek, ile_watkow_na_blok_max);
	przydzielacz_prac przydzielacz_absorbcja(spacer.trwale.liczba_absorberow(), ile_prac_na_watek, ile_watkow_na_blok_max);

	gpu_P_sumator<towar> sumator_P(spacer.iteracjaA.wartosci.rozmiar, 300, 20);
	gpu_sumator<double> sumator(spacer.iteracjaA.wartosci_zaabsorbowane.rozmiar, 300, 20);

	for(uint32_t i = 0; i < liczba_iteracji; i++){		
		//Wskazniki poprawne na GPU
		spacer::dane_iteracji<towar>* iteracja_z = &(spacer.lokalizacja_na_device->iteracjaA);
		spacer::dane_iteracji<towar>* iteracja_do = &(spacer.lokalizacja_na_device->iteracjaB);
		if (spacer.A == false) {
			iteracja_z = &(spacer.lokalizacja_na_device->iteracjaB);
			iteracja_do = &(spacer.lokalizacja_na_device->iteracjaA);
		}
		spacer::dane_trwale<transformata>* trwale = &(spacer.lokalizacja_na_device->trwale);

		iteracja_na_gpu<towar, transformata>start_kernel(przydzielacz_iteracja, 0, spacer.stream_iteracja)(
			trwale, iteracja_z, iteracja_do, przydzielacz_iteracja);
		if(i % co_ile_normalizuj == 0) {
			sumator_P.sumuj(spacer.iteracja_z()->wartosci, &(iteracja_do->prawdopodobienstwo_poprzedniej), spacer.stream_normalizacja);
			sumator.sumuj(spacer.iteracja_z()->wartosci_zaabsorbowane, &(iteracja_do->zaabsorbowane_poprzedniej), spacer.stream_normalizacja);
			policz_wspolczynnik_normalizacji<towar><<<1, 1, 0, spacer.stream_normalizacja>>>(iteracja_z, iteracja_do, spacer.trwale.poczatkowe_prawdopodobienstwo);
		} else {
			nie_normalizuj<towar><<<1, 1, 0, spacer.stream_normalizacja>>>(iteracja_z, iteracja_do);
		}
		if(i % co_ile_zapisac == 0){
			spacer.zapisz_iteracje_z_cuda();
		}

		if(i % co_ile_absorbuj == 0){
			absorbuj_na_gpu<towar, transformata>start_kernel(przydzielacz_absorbcja, 0, spacer.stream_iteracja)(
				trwale, iteracja_z, iteracja_do, 1.0, przydzielacz_absorbcja);
		} else {
			absorbuj_na_gpu<towar, transformata>start_kernel(przydzielacz_absorbcja, 0, spacer.stream_iteracja)(
				trwale, iteracja_z, iteracja_do, 0.0, przydzielacz_absorbcja);
		}
		spacer.dokoncz_iteracje(delta_t);
		zakoncz_iteracje<towar, transformata><<<1, 1, 0, spacer.stream_iteracja>>>(iteracja_do, spacer.iteracja_z()->czas);

		sprawdzCudaErrors(cudaStreamSynchronize(spacer.stream_iteracja));
		sprawdzCudaErrors(cudaStreamSynchronize(spacer.stream_normalizacja));
		sprawdzCudaErrors(cudaStreamSynchronize(spacer.stream_pamiec_operacje));
		sprawdzCudaErrors(cudaGetLastError());

	}
}

template __host__ void iteracje_na_gpu<zesp, TMDQ>(spacer_losowy<zesp, TMDQ>& spacer, double delta_t,
	uint64_t liczba_iteracji, uint64_t ile_prac_na_watek, uint32_t ile_watkow_na_blok_max, uint32_t co_ile_zapisac, uint32_t co_ile_normalizuj, uint32_t co_ile_absorbuj);

template __host__ void iteracje_na_gpu<zesp, TMCQ>(spacer_losowy<zesp, TMCQ>& spacer, double delta_t,
	uint64_t liczba_iteracji, uint64_t ile_prac_na_watek, uint32_t ile_watkow_na_blok_max, uint32_t co_ile_zapisac, uint32_t co_ile_normalizuj, uint32_t co_ile_absorbuj);

__HD__ void testy_macierzy() {
	transformata_macierz<double> t1(1.0);
	transformata_macierz<double> t2(0.75, 0.5, 0.25, 0.5);
	transformata_macierz<double> t3((uint8_t)6);
	double temp[4] = { 0.5, 0.75, 0.5, 0.25 };
	transformata_macierz<double> t4((uint8_t)2, temp);
	transformata_macierz<double> t5(t2);
	transformata_macierz<double> t6 = t3;
	transformata_macierz<double> t7(mnoz(t4, t5));
	transformata_macierz<double> t8(tensor(t4, t5));
	IF_HOST(printf("t8: %s\n", t8.str().c_str());)
	bool t9 = (t6 == t3);
}
__device__ void cuda_test2() {
	printf("testuje2\n");
	testy_macierzy();
}

__global__ void cuda_test(){
	printf("testuje\n");
	testy_macierzy();
	cuda_test2();
}

__host__ void cuda_tester(){
	cudaEvent_t start, stop;
	sprawdzCudaErrors(cudaEventCreate(&start));
	sprawdzCudaErrors(cudaEventCreate(&stop));

	cudaStream_t stream; // stream jest konieczny do printowania
	sprawdzCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	cuda_test<<<16, 1, 0,stream>>>();

	// Record the stop event
	sprawdzCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	sprawdzCudaErrors(cudaEventSynchronize(stop));

	sprawdzCudaErrors(cudaStreamSynchronize(stream));

	sprawdzCudaErrors(cudaEventDestroy(start));
	sprawdzCudaErrors(cudaEventDestroy(stop));
	sprawdzCudaErrors(cudaStreamDestroy(stream));
}

#ifdef __CUDA_ARCH__
#include "transformaty.cu"
#include "zesp.cu"
#endif

