#include "spacer_losowy.h"

#include "transformaty_wyspecializowane.h"

//#include "zesp.h"

template <typename towar, typename transformata>
__global__ void iteracje_na_gpu(spacer_losowy<towar, transformata>* lokalizacja_na_device, uint64_t ile_blokow,
								uint64_t liczba_iteracji, uint64_t ile_prac_wykonac) {
	spacer::dane_trwale<transformata>& trwale = lokalizacja_na_device->trwale;
	for(uint64_t i = 0; i < liczba_iteracji; i++){
		uint64_t index_wierzcholka = 0;

		spacer::dane_iteracji<towar>* iteracja_z = &lokalizacja_na_device->iteracjaA;
		spacer::dane_iteracji<towar>* iteracja_do = &lokalizacja_na_device->iteracjaB;
		if (lokalizacja_na_device->A == false) {
			iteracja_z = &lokalizacja_na_device->iteracjaB;
			iteracja_do = &lokalizacja_na_device->iteracjaA;
		}

		for(uint64_t j = 0; j < ile_prac_wykonac; j++){
			uint64_t index_watka = (blockIdx.x * ile_blokow + threadIdx.x) * ile_prac_wykonac + j;

			index_wierzcholka = trwale.znajdz_wierzcholek(index_watka, index_wierzcholka);
			if(index_wierzcholka == (uint64_t)-1){
				//printf("Nadmierny watek: %d", threadIdx.x);
				break;
			}
	
			uint64_t index_w_wierzcholku = trwale.znajdywacz_wierzcholka[index_wierzcholka] - index_watka - 1;
			spacer::wierzcholek& wierzcholek = trwale.wierzcholki[index_wierzcholka];
			trwale.transformaty[wierzcholek.transformer].transformuj(trwale, wierzcholek, *iteracja_z, *iteracja_do, index_w_wierzcholku);
		}
		__syncthreads();
		if(threadIdx.x == 0 && blockIdx.x == 0) lokalizacja_na_device->dokoncz_iteracje(1.0);
		//printf("CL: %d\n", threadIdx.x);
		__syncthreads();
	}
}

__HD__ double dot(const estetyczny_wektor<double>& a, const estetyczny_wektor<double>& b) {
	ASSERT_Z_ERROR_MSG(a.rozmiar == b.rozmiar, "Dot product na innej ilosci elementow\n");
	double sum = zero(double());
	for (uint64_t i = 0; i < a.rozmiar; i++) {
		sum += (a[i] * b[i]);
	}
	return sum;
}

__HD__ zesp dot(const estetyczny_wektor<zesp>& a, const estetyczny_wektor<zesp>& b) {
	ASSERT_Z_ERROR_MSG(a.rozmiar == b.rozmiar, "Dot product na innej ilosci elementow\n");
	zesp sum = zero(zesp());
	for (uint64_t i = 0; i < a.rozmiar; i++) {
		sum += (a[i].sprzezenie() * b[i]);
	}
	return sum;
}

template <typename towar, typename transformata>
__global__ void iteracja_na_gpu(spacer_losowy<towar, transformata>* lokalizacja_na_device, uint64_t ile_blokow) {
	spacer::dane_trwale<transformata>& trwale = lokalizacja_na_device->trwale;
	uint64_t index_wierzcholka = 0;

	spacer::dane_iteracji<towar>* iteracja_z = &lokalizacja_na_device->iteracjaA;
	spacer::dane_iteracji<towar>* iteracja_do = &lokalizacja_na_device->iteracjaB;
	if (lokalizacja_na_device->A == false) {
		iteracja_z = &lokalizacja_na_device->iteracjaB;
		iteracja_do = &lokalizacja_na_device->iteracjaA;
	}

	constexpr int ile_prac_wykonac = 10;

	for(uint64_t j = 0; j < ile_prac_wykonac; j++){
		uint64_t index_watka = (blockIdx.x * ile_blokow + threadIdx.x) * ile_prac_wykonac + j;

		index_wierzcholka = trwale.znajdz_wierzcholek(index_watka, index_wierzcholka);
		if(index_wierzcholka == (uint64_t)-1) break;

		uint64_t index_w_wierzcholku = trwale.znajdywacz_wierzcholka[index_wierzcholka] - index_watka - 1;
		spacer::wierzcholek& wierzcholek = trwale.wierzcholki[index_wierzcholka];
		trwale.transformaty[wierzcholek.transformer].transformuj(trwale, wierzcholek, *iteracja_z, *iteracja_do, index_w_wierzcholku);
	}
	__syncthreads();
	if(threadIdx.x == 0 && blockIdx.x == 0) lokalizacja_na_device->dokoncz_iteracje(1.0);
	__syncthreads();
}

template <typename towar, typename transformata>
__host__ void symulowana_iteracja_na_gpu(spacer_losowy<towar, transformata>* lokalizacja_na_device, uint64_t threadIdx) {
	spacer::dane_trwale<transformata>& trwale = lokalizacja_na_device->trwale;
	uint64_t index_wierzcholka = 0;

	spacer::dane_iteracji<towar>* iteracja_z = &lokalizacja_na_device->iteracjaA;
	spacer::dane_iteracji<towar>* iteracja_do = &lokalizacja_na_device->iteracjaB;
	if (lokalizacja_na_device->A == false) {
		iteracja_z = &lokalizacja_na_device->iteracjaB;
		iteracja_do = &lokalizacja_na_device->iteracjaA;
	}

	constexpr int ile_prac_wykonac = 10;

	for (uint64_t j = 0; j < ile_prac_wykonac; j++) {
		uint64_t index_watka = threadIdx * ile_prac_wykonac + j;

		index_wierzcholka = trwale.znajdz_wierzcholek(index_watka, index_wierzcholka);
		if (index_wierzcholka == (uint64_t)-1) break;

		uint64_t index_w_wierzcholku = trwale.znajdywacz_wierzcholka[index_wierzcholka] - index_watka - 1;
		spacer::wierzcholek& wierzcholek = trwale.wierzcholki[index_wierzcholka];
		trwale.transformaty[wierzcholek.transformer].transformuj(trwale, wierzcholek, *iteracja_z, *iteracja_do, index_w_wierzcholku);
	} 
	//if (threadIdx == 0) lokalizacja_na_device->dokoncz_iteracje(1.0);
}

template __host__ void symulowana_iteracja_na_gpu<zesp, TMDQ>(spacer_losowy<zesp, TMDQ>* lokalizacja_na_device, uint64_t threadIdx);

template <typename towar, typename transformata>
__host__ void iteruj_na_gpu(spacer_losowy<towar, transformata>& spacer,
	uint64_t liczba_iteracji) {

	//spacer.trwale.ile_watkow(10)
	uint64_t ile_prac = spacer.trwale.ile_prac();
	constexpr int max_ilosc_watkow_w_bloku = 100;
	uint64_t ile_prac_na_watek = ile_prac / max_ilosc_watkow_w_bloku + 1;
	iteracje_na_gpu<towar, transformata><<<1, max_ilosc_watkow_w_bloku, 0, spacer.stream>>>(spacer.lokalizacja_na_device, 1, liczba_iteracji, ile_prac_na_watek);
	checkCudaErrors(cudaStreamSynchronize(spacer.stream));
	checkCudaErrors(cudaGetLastError());
}

template __host__ void iteruj_na_gpu<zesp, TMDQ>(spacer_losowy<zesp, TMDQ>& spacer,
	uint64_t liczba_iteracji);


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
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	cudaStream_t stream; // stream jest konieczny do printowania
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	cuda_test<<<16, 1, 0,stream>>>();

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	checkCudaErrors(cudaStreamSynchronize(stream));

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaStreamDestroy(stream));
}

#ifdef __CUDA_ARCH__
#include "transformaty.cu"
#include "zesp.cu"
#endif

