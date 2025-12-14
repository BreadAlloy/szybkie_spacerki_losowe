#include "spacer_losowy.h"

#include "transformaty_wyspecializowane.h"

//#include <cooperative_groups.h> nie mam compute compatibility 9.0 :( cluster.sync()

template <typename towar, typename transformata>
__global__ void iteracje_na_gpu(spacer_losowy<towar, transformata>* lokalizacja_na_device, uint64_t ile_watkow,
								uint64_t liczba_iteracji, uint64_t ile_prac_wykonac) {

	
	//__shared__ zesp* dzielone[300*300*4];

	spacer::dane_trwale<transformata>& trwale = lokalizacja_na_device->trwale;
	for(uint64_t i = 0; i < liczba_iteracji; i++){

		spacer::dane_iteracji<towar>* iteracja_z = &lokalizacja_na_device->iteracjaA;
		spacer::dane_iteracji<towar>* iteracja_do = &lokalizacja_na_device->iteracjaB;
		if (lokalizacja_na_device->A == false) {
			iteracja_z = &lokalizacja_na_device->iteracjaB;
			iteracja_do = &lokalizacja_na_device->iteracjaA;
		}

		for(uint64_t j = 0; j < ile_prac_wykonac; j++){
			uint64_t index_pracownika = threadIdx.x + ile_watkow * j;

			spacer::info_pracownika IP = trwale.znajdz_wierzcholek(index_pracownika);

			uint32_t index_wierzcholka = IP.index_wierzcholka;
			uint8_t index_w_wierzcholku = IP.index_w_wierzcholku;

			if((index_wierzcholka == (uint32_t)-1) || (index_w_wierzcholku == (uint8_t)-1)){
				//printf("Nadmierny watek: %d", threadIdx.x);
				break;
			}
	
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
__global__ void iteracja_na_gpu(spacer_losowy<towar, transformata>* lokalizacja_na_device, uint64_t ile_watkow_na_blok, uint64_t ile_blokow, uint64_t ile_prac_wykonac) {
	spacer::dane_trwale<transformata>& trwale = lokalizacja_na_device->trwale;

	spacer::dane_iteracji<towar>* iteracja_z = &lokalizacja_na_device->iteracjaA;
	spacer::dane_iteracji<towar>* iteracja_do = &lokalizacja_na_device->iteracjaB;
	if (lokalizacja_na_device->A == false) {
		iteracja_z = &lokalizacja_na_device->iteracjaB;
		iteracja_do = &lokalizacja_na_device->iteracjaA;
	}

	for(uint64_t j = 0; j < ile_prac_wykonac; j++){
		uint64_t index_pracownika = ile_watkow_na_blok * (ile_prac_wykonac * blockIdx.x + j) + threadIdx.x;
	
		spacer::info_pracownika IP = trwale.znajdz_wierzcholek(index_pracownika);

		uint32_t index_wierzcholka = IP.index_wierzcholka;
		uint8_t index_w_wierzcholku = IP.index_w_wierzcholku;

		if ((index_wierzcholka == (uint32_t)-1) || (index_w_wierzcholku == (uint8_t)-1)) {
			//printf("Nadmierny watek: %d", threadIdx.x);
			break;
		}

		spacer::wierzcholek& wierzcholek = trwale.wierzcholki[index_wierzcholka];

		trwale.transformaty[wierzcholek.transformer].transformuj(trwale, wierzcholek, *iteracja_z, *iteracja_do, index_w_wierzcholku);
	}
}

template <typename towar, typename transformata>
__global__ void absorbuj_na_gpu(spacer_losowy<towar, transformata>* lokalizacja_na_device, uint64_t ile_watkow_na_blok, uint64_t ile_blokow, uint64_t ile_prac_wykonac){
	spacer::dane_trwale<transformata>& trwale = lokalizacja_na_device->trwale;

	spacer::dane_iteracji<towar>* iteracja_z = &lokalizacja_na_device->iteracjaA;
	spacer::dane_iteracji<towar>* iteracja_do = &lokalizacja_na_device->iteracjaB;
	if (lokalizacja_na_device->A == false) {
		iteracja_z = &lokalizacja_na_device->iteracjaB;
		iteracja_do = &lokalizacja_na_device->iteracjaA;
	}

	for (uint64_t j = 0; j < ile_prac_wykonac; j++) {
		uint64_t index_pracownika = ile_watkow_na_blok * (ile_prac_wykonac * blockIdx.x + j) + threadIdx.x;

		if(index_pracownika >= trwale.liczba_absorberow()){
			break;
		}
		uint64_t indeks_absorbowany = trwale.indeksy_absorbowane[index_pracownika];
		towar zaabsorbowane = iteracja_do->wartosci[indeks_absorbowany];
		iteracja_do->wartosci[indeks_absorbowany] = zero(towar());
		iteracja_do->wartosci_zaabsorbowane[index_pracownika] = P(zaabsorbowane) + iteracja_z->wartosci_zaabsorbowane[index_pracownika];
	}
}

template <typename towar, typename transformata>
__global__ void zakoncz_iteracje(spacer_losowy<towar, transformata>* lokalizacja_na_device){ //na jeden watek w jednym bloku
	lokalizacja_na_device->dokoncz_iteracje(1.0);
}

template <typename towar, typename transformata>
__host__ void iteruj_na_gpu(spacer_losowy<towar, transformata>& spacer,
	uint64_t liczba_iteracji, uint64_t liczba_watkow) {

	uint64_t ile_prac = spacer.trwale.ile_prac();
	uint64_t ile_prac_na_watek = ile_prac / liczba_watkow + 1;
	iteracje_na_gpu<towar, transformata><<<1, liczba_watkow, 0, spacer.stream>>>(spacer.lokalizacja_na_device, liczba_watkow, liczba_iteracji, ile_prac_na_watek);
	checkCudaErrors(cudaStreamSynchronize(spacer.stream));
	checkCudaErrors(cudaGetLastError());
}

template __host__ void iteruj_na_gpu<zesp, TMDQ>(spacer_losowy<zesp, TMDQ>& spacer,
	uint64_t liczba_iteracji, uint64_t liczba_watkow);

template <typename towar, typename transformata>
__host__ void iteracje_na_gpu(spacer_losowy<towar, transformata>& spacer,
	uint64_t liczba_iteracji, uint64_t ile_prac_na_watek, uint32_t ile_watkow_na_blok_max, uint32_t co_ile_zapisac) {

	uint64_t ile_prac = spacer.trwale.ile_prac();
	uint64_t ile_watkow_sumarycznie = ile_prac / ile_prac_na_watek + 1;
	uint64_t ile_blokow = ile_watkow_sumarycznie / ile_watkow_na_blok_max + 1;
	uint64_t ile_watkow = ile_watkow_sumarycznie / ile_blokow + 1;

	uint64_t ile_prac_absorbcja = spacer.trwale.liczba_absorberow();
	uint64_t ile_watkow_sumarycznie_absorbcja = ile_prac_absorbcja / ile_prac_na_watek + 1;
	uint64_t ile_blokow_absorbcja = ile_watkow_sumarycznie_absorbcja / ile_watkow_na_blok_max + 1;
	uint64_t ile_watkow_absorbcja = ile_watkow_sumarycznie_absorbcja / ile_blokow_absorbcja + 1;

	for(uint32_t i = 0; i < liczba_iteracji; i++){
		iteracja_na_gpu<towar, transformata><<<ile_blokow, ile_watkow, 0, spacer.stream>>>(spacer.lokalizacja_na_device, ile_watkow, ile_blokow, ile_prac_na_watek);
		if(i % co_ile_zapisac == 0){
			spacer.zapisz_iteracje_z_cuda();
		}
		checkCudaErrors(cudaStreamSynchronize(spacer.stream));
		absorbuj_na_gpu<towar, transformata><<<ile_blokow_absorbcja, ile_watkow_absorbcja, 0, spacer.stream >> > (spacer.lokalizacja_na_device, ile_watkow_absorbcja, ile_blokow_absorbcja, ile_prac_na_watek);
		spacer.dokoncz_iteracje(1.0);
		checkCudaErrors(cudaStreamSynchronize(spacer.stream));
		zakoncz_iteracje<towar, transformata><<<1, 1, 0, spacer.stream>>>(spacer.lokalizacja_na_device);
		checkCudaErrors(cudaStreamSynchronize(spacer.stream));
		checkCudaErrors(cudaGetLastError());
	}
}

template __host__ void iteracje_na_gpu<zesp, TMDQ>(spacer_losowy<zesp, TMDQ>& spacer,
	uint64_t liczba_iteracji, uint64_t ile_prac_na_watek, uint32_t ile_watkow_na_blok_max, uint32_t co_ile_zapisac);

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

