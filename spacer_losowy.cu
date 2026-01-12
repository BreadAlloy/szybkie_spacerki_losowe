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

			trwale.transformaty[wierzcholek.transformer].transformuj(trwale, wierzcholek, *iteracja_z, *iteracja_do, index_w_wierzcholku, index_wierzcholka);
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

		trwale.transformaty[wierzcholek.transformer].transformuj(trwale, wierzcholek, *iteracja_z, *iteracja_do, index_w_wierzcholku, index_wierzcholka);
	}
}

template <typename towar, typename transformata>
__global__ void absorbuj_na_gpu(spacer_losowy<towar, transformata>* lokalizacja_na_device, double procent_absorbowany,
	uint64_t ile_watkow_na_blok, uint64_t ile_blokow, uint64_t ile_prac_wykonac){
	spacer::dane_trwale<transformata>& trwale = lokalizacja_na_device->trwale;

	spacer::dane_iteracji<towar>* iteracja_z = &lokalizacja_na_device->iteracjaA;
	spacer::dane_iteracji<towar>* iteracja_do = &lokalizacja_na_device->iteracjaB;
	if (lokalizacja_na_device->A == false) {
		iteracja_z = &lokalizacja_na_device->iteracjaB;
		iteracja_do = &lokalizacja_na_device->iteracjaA;
	}

	double norma_zabranego = NORMA(1.0, procent_absorbowany, towar());
	double norma_pozostawionego = NORMA(1.0, 1.0 - procent_absorbowany, towar());

	for (uint64_t j = 0; j < ile_prac_wykonac; j++) {
		uint64_t index_pracownika = ile_watkow_na_blok * (ile_prac_wykonac * blockIdx.x + j) + threadIdx.x;

		if(index_pracownika >= trwale.liczba_absorberow()){
			break;
		}
		uint64_t indeks_absorbowany = trwale.indeksy_absorbowane[index_pracownika];
		towar zaabsorbowane = iteracja_do->wartosci[indeks_absorbowany] * norma_zabranego;
		iteracja_do->wartosci[indeks_absorbowany] *= norma_pozostawionego;
		iteracja_do->wartosci_zaabsorbowane[index_pracownika] = P(zaabsorbowane) + iteracja_z->wartosci_zaabsorbowane[index_pracownika];
	}
}

// na jeden blok
template <typename towar>
__global__ void suma_prawdopodobienstwa_gpu(statyczny_wektor<towar>* wektor, double* gdzie_zapisac, const uint64_t ile_watkow, const uint64_t ile_prac_wykonac) {
	__shared__ double wspolne[1024];

	double wysumowane = 0.0;
	for (uint64_t j = 0; j < ile_prac_wykonac; j++) {
		uint64_t index_pracownika = ile_watkow * j + threadIdx.x; // moze da sie zrobic lepsza kolejnosc

		if (index_pracownika >= wektor->rozmiar) {
			break;
		}
		wysumowane += P(wektor->operator[](index_pracownika));
	}
	wspolne[threadIdx.x] = wysumowane;

	__syncthreads();
	if(threadIdx.x == 0){
		double finalna_suma = 0.0;
		for(uint64_t i = 0; i < ile_watkow; i++){
			finalna_suma += wspolne[i];
		}
		(*gdzie_zapisac) = finalna_suma;
	}
}

// na jeden blok
template <typename towar>
__global__ void suma_gpu(statyczny_wektor<towar>* wektor, double* gdzie_zapisac,
	const uint64_t ile_watkow, const uint64_t ile_prac_wykonac) {
	__shared__ double wspolne[1024];

	double wysumowane = 0.0;
	for (uint64_t j = 0; j < ile_prac_wykonac; j++) {
		uint64_t index_pracownika = ile_watkow * j + threadIdx.x; // moze da sie zrobic lepsza kolejnosc

		if (index_pracownika >= wektor->rozmiar) {
			break;
		}
		wysumowane += wektor->operator[](index_pracownika);
	}
	wspolne[threadIdx.x] = wysumowane;

	__syncthreads();
	if (threadIdx.x == 0) {
		double finalna_suma = 0.0;
		for (uint64_t i = 0; i < ile_watkow; i++) {
			finalna_suma += wspolne[i];
		}
		(*gdzie_zapisac) = finalna_suma;
	}
}

template <typename towar, typename transformata>
__host__ void przygotuj_do_normalizacji(spacer_losowy<towar, transformata>& spacer, uint64_t ile_pracy_na_normalizatora){
	spacer::dane_iteracji<towar>* iteracja_z = &spacer.iteracjaA;
	spacer::dane_iteracji<towar>* iteracja_do = &spacer.iteracjaB;
	spacer::dane_iteracji<towar>* device_iteracja_z = &(spacer.lokalizacja_na_device->iteracjaA);
	spacer::dane_iteracji<towar>* device_iteracja_do = &(spacer.lokalizacja_na_device->iteracjaB);
	if (spacer.A == false) {
		iteracja_z = &spacer.iteracjaB;
		iteracja_do = &spacer.iteracjaA;
		device_iteracja_z = &(spacer.lokalizacja_na_device->iteracjaB);
		device_iteracja_do = &(spacer.lokalizacja_na_device->iteracjaA);
	}
	
	uint64_t ile_sumy_prawdopodobienstwa = iteracja_z->wartosci.rozmiar;
	uint64_t ile_sumy_zaabsorbowanego = iteracja_z->wartosci_zaabsorbowane.rozmiar;

	uint64_t ile_sumatorow_prawdopodobienstwa = ile_sumy_prawdopodobienstwa / ile_pracy_na_normalizatora + 1L;
	uint64_t ile_sumatorow_zaabsorbowanego = ile_sumy_zaabsorbowanego / ile_pracy_na_normalizatora + 1L;

	suma_prawdopodobienstwa_gpu<towar><<<1, (uint32_t)ile_sumatorow_prawdopodobienstwa, 0, spacer.stream>>>(
		&(device_iteracja_z->wartosci), &(device_iteracja_do->prawdopodobienstwo_poprzedniej),
		ile_sumatorow_prawdopodobienstwa, ile_pracy_na_normalizatora
	);

	suma_gpu<double><<<1, (uint32_t)ile_sumatorow_zaabsorbowanego, 0, spacer.stream>>>(
		&(device_iteracja_z->wartosci_zaabsorbowane), &(device_iteracja_do->zaabsorbowane_poprzedniej),
		ile_sumatorow_zaabsorbowanego, ile_pracy_na_normalizatora
	);

}

template <typename towar, typename transformata>
__global__ void policz_wspolczynnik_normalizacji(spacer_losowy<towar, transformata>* lokalizacja_na_device) { //na jeden watek w jednym bloku
	spacer::dane_iteracji<towar>* iteracja_z = &lokalizacja_na_device->iteracjaA;
	spacer::dane_iteracji<towar>* iteracja_do = &lokalizacja_na_device->iteracjaB;
	if (lokalizacja_na_device->A == false) {
		iteracja_z = &lokalizacja_na_device->iteracjaB;
		iteracja_do = &lokalizacja_na_device->iteracjaA;
	}

	double powinno_byc = lokalizacja_na_device->trwale.poczatkowe_prawdopodobienstwo - iteracja_do->zaabsorbowane_poprzedniej;
	iteracja_do->norma_poprzedniej_iteracji = NORMA(iteracja_do->prawdopodobienstwo_poprzedniej, powinno_byc, towar());
}

template <typename towar, typename transformata>
__global__ void zakoncz_iteracje(spacer_losowy<towar, transformata>* lokalizacja_na_device, double delta_t) { //na jeden watek w jednym bloku
	lokalizacja_na_device->dokoncz_iteracje(delta_t);
}

template <typename towar, typename transformata>
__global__ void nie_normalizuj(spacer_losowy<towar, transformata>* lokalizacja_na_device) { //na jeden watek w jednym bloku
	lokalizacja_na_device->nie_normalizuj();
}

template <typename towar, typename transformata>
__host__ void iteruj_na_gpu(spacer_losowy<towar, transformata>& spacer,
	uint64_t liczba_iteracji, uint64_t liczba_watkow) {

	uint64_t ile_prac = spacer.trwale.ile_prac();
	uint64_t ile_prac_na_watek = ile_prac / liczba_watkow + 1;
	iteracje_na_gpu<towar, transformata><<<1, (uint32_t)liczba_watkow, 0, spacer.stream>>>(spacer.lokalizacja_na_device, liczba_watkow, liczba_iteracji, ile_prac_na_watek);
	checkCudaErrors(cudaStreamSynchronize(spacer.stream));
	checkCudaErrors(cudaGetLastError());
}

template __host__ void iteruj_na_gpu<zesp, TMDQ>(spacer_losowy<zesp, TMDQ>& spacer,
	uint64_t liczba_iteracji, uint64_t liczba_watkow);

template <typename towar, typename transformata>
__host__ void iteracje_na_gpu(spacer_losowy<towar, transformata>& spacer, double delta_t,
	uint64_t liczba_iteracji, uint64_t ile_prac_na_watek, uint32_t ile_watkow_na_blok_max, uint32_t co_ile_zapisac, uint32_t co_ile_normalizuj, uint32_t co_ile_absorbuj) {

	uint64_t ile_prac = spacer.trwale.ile_prac();
	uint64_t ile_watkow_sumarycznie = ile_prac / ile_prac_na_watek + 1;
	uint64_t ile_blokow = ile_watkow_sumarycznie / ile_watkow_na_blok_max + 1;
	uint64_t ile_watkow = ile_watkow_sumarycznie / ile_blokow + 1;

	uint64_t ile_prac_absorbcja = spacer.trwale.liczba_absorberow();
	uint64_t ile_watkow_sumarycznie_absorbcja = ile_prac_absorbcja / ile_prac_na_watek + 1;
	uint64_t ile_blokow_absorbcja = ile_watkow_sumarycznie_absorbcja / ile_watkow_na_blok_max + 1;
	uint64_t ile_watkow_absorbcja = ile_watkow_sumarycznie_absorbcja / ile_blokow_absorbcja + 1;

	for(uint32_t i = 0; i < liczba_iteracji; i++){
		
		iteracja_na_gpu<towar, transformata><<<(uint32_t)ile_blokow, (uint32_t)ile_watkow, 0, spacer.stream>>>(spacer.lokalizacja_na_device, ile_watkow, ile_blokow, ile_prac_na_watek);
		if(i % co_ile_normalizuj == 0) {
			przygotuj_do_normalizacji<towar, transformata>(spacer, spacer.trwale.gdzie_wyslac.rozmiar / 700UL + 1UL);
		} else {
			nie_normalizuj<towar, transformata><<<1, 1, 0, spacer.stream>>>(spacer.lokalizacja_na_device);
		}
		if(i % co_ile_zapisac == 0){
			spacer.zapisz_iteracje_z_cuda();
		}
		checkCudaErrors(cudaStreamSynchronize(spacer.stream));
		checkCudaErrors(cudaGetLastError());

		if(i % co_ile_absorbuj == 0){
			absorbuj_na_gpu<towar, transformata><<<(uint32_t)ile_blokow_absorbcja, (uint32_t)ile_watkow_absorbcja, 0, spacer.stream>>> (
			spacer.lokalizacja_na_device, 1.0, ile_watkow_absorbcja, ile_blokow_absorbcja, ile_prac_na_watek);
		} else {
			absorbuj_na_gpu<towar, transformata><<<(uint32_t)ile_blokow_absorbcja, (uint32_t)ile_watkow_absorbcja, 0, spacer.stream>>> (
			spacer.lokalizacja_na_device, 0.0, ile_watkow_absorbcja, ile_blokow_absorbcja, ile_prac_na_watek);
		}
		if (i % co_ile_normalizuj == 0) {
			policz_wspolczynnik_normalizacji<towar, transformata><<<1, 1, 0, spacer.stream>>>(spacer.lokalizacja_na_device);
		}
		checkCudaErrors(cudaStreamSynchronize(spacer.stream));
		checkCudaErrors(cudaGetLastError());
		
		zakoncz_iteracje<towar, transformata><<<1, 1, 0, spacer.stream>>>(spacer.lokalizacja_na_device, delta_t);
		spacer.dokoncz_iteracje(delta_t);
		checkCudaErrors(cudaStreamSynchronize(spacer.stream));
		checkCudaErrors(cudaGetLastError());
	}
	//checkCudaErrors(cudaStreamSynchronize(spacer.stream));
	//checkCudaErrors(cudaGetLastError());
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

