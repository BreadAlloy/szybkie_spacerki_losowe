#pragma once

#include "graf.h"

#include "wektory.h"

#include "transformaty.h"

#include "zesp.h"

typedef double typ_prawdopodobienstwa;

static inline __HD__ double P(double a){
	return a;
}

static inline __HD__ double P(const zesp& a) {
	return a.norm();
}

constexpr typ_prawdopodobienstwa dokladnosc = 1.0e-10;
constexpr typ_prawdopodobienstwa tolerancja = 1.0e-10;

namespace spacer{

struct wierzcholek{
	uint64_t start_wartosci = (uint64_t)(-1);
	uint64_t transformer = (uint64_t)(-1);
	uint8_t liczba_kierunkow = 0; //tyle w¹tków potrzeba

	__HD__ wierzcholek(uint64_t start_wartosci = (uint64_t)-1, uint8_t liczba_kierunkow = 0)
	: start_wartosci(start_wartosci), liczba_kierunkow(liczba_kierunkow){}

};

struct info_pracownika {
	uint32_t index_wierzcholka;
	uint8_t index_w_wierzcholku;

	__HD__ info_pracownika(uint32_t index_wierzcholka, uint8_t index_w_wierzcholku)
		: index_w_wierzcholku(index_w_wierzcholku), index_wierzcholka(index_wierzcholka) {
	}

	__HD__ bool operator==(info_pracownika drugi) {
		return (index_wierzcholka == drugi.index_wierzcholka) &&
			(index_w_wierzcholku == drugi.index_w_wierzcholku);
	}
};

typedef info_pracownika indeks_pozycji;
typedef std::vector<indeks_pozycji> indeksy_pozycji;

template<typename transformata>
struct uklad_transformat{ // zbiornik na transformaty potem bêdzie sprawdzane czy dzia³aj¹ w parze z danymi trwalymi
	std::vector<uint64_t> indeksy; // tyle co wierzcholkow
	std::vector<transformata> transformery;

	__host__ uklad_transformat(uint64_t ile_wierzcholkow)
	: indeksy(ile_wierzcholkow){}

	__host__ uint64_t dodaj_transformate(transformata& T){
		transformery.push_back(T);
		return transformery.size() - 1;
	}

	__host__ void podepnij_transformate(uint64_t id_transformaty, uint64_t id_wierzcholka){
		ASSERT_Z_ERROR_MSG(id_transformaty < transformery.size(), "Nie ma transformaty o takim indeksie\n");
		indeksy[id_wierzcholka] = id_transformaty;
	}

	__host__ transformata& jaka_transformata_na(uint64_t index_wierzcholka){
		return transformery[indeksy[index_wierzcholka]];
	}
};

//IF_HD(
//const info_pracownika niepotrzebny_pracownik((uint32_t)-1, (uint8_t)-1);,
//__constant__ info_pracownika niepotrzebny_pracownik((uint32_t)-1, (uint8_t)-1);
//)

template<typename transformata>
struct dane_trwale{ //operatory, to gdzie wysy³aæ, przestrzen, raczej nie zmienia sie z czasem
	statyczny_wektor<uint64_t> gdzie_wyslac; // krawedzie w grafie
	statyczny_wektor<spacer::wierzcholek> wierzcholki;
	statyczny_wektor<info_pracownika> znajdywacz_wierzcholka; // znajduje wierzcholek na podstawie indexu watka
	statyczny_wektor<transformata> transformaty;
	statyczny_wektor<uint64_t> indeksy_absorbowane;

	__host__ dane_trwale(const graf& przestrzen) {
		ASSERT_Z_ERROR_MSG(przestrzen.czy_gotowy(), "graf nie byl gotowy\n");

		wierzcholki.malloc(przestrzen.liczba_wierzcholkow());

		uint64_t suma_kubelkow = 0;
		for (ID_W i = 0; i < przestrzen.liczba_wierzcholkow(); i++) {
			uint8_t liczba_kierunkow = przestrzen.wierzcholki[i].liczba_polaczen();
			wierzcholki[i] = spacer::wierzcholek(suma_kubelkow, liczba_kierunkow);
			suma_kubelkow += liczba_kierunkow;
		}

		// teraz suma_kubelkow jest równy sumie liczby kube³ków
		gdzie_wyslac.malloc(suma_kubelkow);

		for (size_t i = 0; i < przestrzen.liczba_krawedzi(); i++) {
			grafowe::krawedz k = przestrzen.krawedzie[i];
			size_t index_z = wierzcholki[k.index_wierzcholka_z].start_wartosci + k.index_kubelka_z;
			size_t index_do = wierzcholki[k.index_wierzcholka_do].start_wartosci + k.index_kubelka_do;
			gdzie_wyslac[index_z] = index_do;
		}
	}

	// mo¿na przygotowac dopiero po dodaniu transformat
	__host__ void przygotuj_znajdywacz_wierzcholka(){
		uint64_t ile_prac = 0;
		for (ID_W i = 0; i < wierzcholki.rozmiar; i++) {
			for (uint8_t j = 0; j < transformaty[(wierzcholki[i]).transformer].ile_watkow; j++) {
				ile_prac++;
			}
		}
		znajdywacz_wierzcholka.malloc(ile_prac);

		uint64_t mem = 0;
		for (ID_W i = 0; i < wierzcholki.rozmiar; i++) {
			for(uint8_t j = 0; j < transformaty[(wierzcholki[i]).transformer].ile_watkow; j++){
				znajdywacz_wierzcholka[mem] = info_pracownika(i, j);
				mem++;
			}
		}
	}

	__HD__ uint64_t ile_watkow(uint64_t ile_prac_na_watek){
		return (ile_prac() / ile_prac_na_watek) + 1;
	}

	__HD__ uint64_t ile_prac(){
		return znajdywacz_wierzcholka.rozmiar;
	}

	__HD__ uint64_t liczba_kubelkow(){
		return gdzie_wyslac.rozmiar;
	}

	__HD__ uint64_t liczba_absorberow(){
		return indeksy_absorbowane.rozmiar;
	}

	__host__ void dodaj_transformaty(uklad_transformat<transformata>& uklad){
		ASSERT_Z_ERROR_MSG(uklad.indeksy.size() == wierzcholki.rozmiar, "jest inna liczba transformat i wierzchoklow\n");
		bool error = false;
		for(uint64_t i = 0; i < wierzcholki.rozmiar; i++){
			if(wierzcholki[i].liczba_kierunkow != uklad.jaka_transformata_na(i).arrnosc){
				error = true;
				printf("Transformata na wierzcholek: %lld o indeksie jest na zla arrnosc. Powinno byc %d, a jest %d\n", i, wierzcholki[i].liczba_kierunkow, uklad.jaka_transformata_na(i).arrnosc);
			}
			wierzcholki[i].transformer = uklad.indeksy[i];
		}
		ASSERT_Z_ERROR_MSG(!error, "Jedna albo wiecej arrnosci sie nie zgadza\n");
		

		error = false; // wlasciwie to nic nie robi
		// czas przes³aæ transformaty
		transformaty.malloc(uklad.transformery.size());
		for(uint64_t i = 0; i < uklad.transformery.size(); i++){
			IF_HOST(std::construct_at(&(transformaty[i]), uklad.transformery[i]);)
		}
	}

	__host__ void dodaj_absorbery(indeksy_pozycji gdzie_absorbery){
		indeksy_absorbowane.malloc(gdzie_absorbery.size());
		for(uint64_t i = 0; i < gdzie_absorbery.size(); i++){
			indeks_pozycji gdzie = gdzie_absorbery[i];
			wierzcholek& w = wierzcholki[gdzie.index_wierzcholka];
			ASSERT_Z_ERROR_MSG(w.liczba_kierunkow > gdzie.index_w_wierzcholku, "Wierzcholek nie ma tylu polaczen\n");
			uint64_t indeks_absorbowany = w.start_wartosci + (uint64_t)gdzie.index_w_wierzcholku;
			for(uint64_t j = 0; j < i; j++){
				ASSERT_Z_ERROR_MSG(indeks_absorbowany != indeksy_absorbowane[j], "Wierzcholkek %d, i kubelek %d sa juz absorbowane\n" SEP gdzie.index_wierzcholka SEP gdzie.index_w_wierzcholku);
			}
			indeksy_absorbowane[i] = indeks_absorbowany;
		}
	}

	__host__ void zbuduj_na_cuda(cudaStream_t stream){
		gdzie_wyslac.cuda_malloc();
		gdzie_wyslac.cuda_zanies(stream);

		wierzcholki.cuda_malloc();
		wierzcholki.cuda_zanies(stream);

		znajdywacz_wierzcholka.cuda_malloc();
		znajdywacz_wierzcholka.cuda_zanies(stream);

		for(uint64_t i = 0; i < transformaty.rozmiar; i++){
			transformaty[i].cuda_malloc();
			transformaty[i].cuda_zanies(stream);
		}

		transformaty.cuda_malloc();
		transformaty.cuda_zanies(stream);

		indeksy_absorbowane.cuda_malloc();
		indeksy_absorbowane.cuda_zanies(stream);
	}

	__host__ void zburz_na_cuda(){
		gdzie_wyslac.cuda_free();
		wierzcholki.cuda_free();
		znajdywacz_wierzcholka.cuda_free();

		for (uint64_t i = 0; i < transformaty.rozmiar; i++) {
			transformaty[i].cuda_free();
		}
		transformaty.cuda_free();
		indeksy_absorbowane.cuda_free();
	}

	__HD__ uint64_t liczba_wierzcholkow(){
		return wierzcholki.rozmiar;
	}

	__HD__ info_pracownika znajdz_wierzcholek(uint64_t index_pracownika, uint64_t cache = 0) {
		// zwraca index szukanego wierzcho³ka
		if(index_pracownika < ile_prac()){
			return znajdywacz_wierzcholka[index_pracownika];
		}
		return info_pracownika((uint32_t)-1, (uint8_t)-1); // nie znaleziono wierzcholka
	}
	
	__host__ bool czy_gotowy(){
		bool ret = true;
		if(transformaty.rozmiar == 0){
			printf("Transformaty w spacerze nie sa gotowe\n");
			ret = false;
		}
		// troche brakuje checku czy zosta³o wywo³ane przygotuj_znajdywacz_wierzcholka()
		return ret;
	}

	__host__ void zamien_transformate(uint32_t index_do_wymiany, transformata& wkladana) {
		ASSERT_Z_ERROR_MSG(index_do_wymiany < transformaty.rozmiar, "Nie ma transformaty o takim indeksie\n");
		transformata& wymieniana = transformaty[index_do_wymiany];
		ASSERT_Z_ERROR_MSG(wymieniana.arrnosc == wkladana.arrnosc, "Wkladana transformata ma inna arrnosc\n");
		wymieniana = wkladana;
	}

	__host__ void odwroc() {
		// dla dowolnego grafu to nie dzia³a, trzeba permutacje odwróciæ
		for(uint32_t i = 0; i < transformaty.rozmiar; i++){
			transformata odwracana = hermituj(transformaty[i]);
			zamien_transformate(i, odwracana);
		}
	}
};

template<typename towar>
struct dane_iteracji{
	statyczny_wektor<towar> wartosci;
	statyczny_wektor<double> wartosci_zaabsorbowane;
	double czas = 0.0;

	dane_iteracji(uint64_t liczba_wartosci = 0, uint64_t liczba_absorberow = 0)
	: wartosci(liczba_wartosci), wartosci_zaabsorbowane(liczba_absorberow) {}

	dane_iteracji(const dane_iteracji& kopiowane)
	: wartosci(kopiowane.wartosci), czas(kopiowane.czas),
	wartosci_zaabsorbowane(kopiowane.wartosci_zaabsorbowane){}

	void zeruj(){
		wartosci.memset(zero(towar()));
		wartosci_zaabsorbowane.memset(zero(double()));
	}

	__host__ typ_prawdopodobienstwa prawdopodobienstwo_suma() const {
		typ_prawdopodobienstwa suma = 0.0;
		for (uint64_t i = 0; i < wartosci.rozmiar; i++) {
			suma += P(wartosci[i]);
		}
		return suma;
	}

	__HD__ towar& operator[](uint64_t index) {
		return wartosci[index];
	}

	__HD__ towar operator[](uint64_t index) const {
		return wartosci[index];
	}

	__host__ void cuda_malloc(){
		wartosci.cuda_malloc();
		wartosci_zaabsorbowane.cuda_malloc();
	}

	__host__ void cuda_zanies(cudaStream_t stream){
		wartosci.cuda_zanies(stream);
		wartosci_zaabsorbowane.cuda_zanies(stream);
	}

	__host__ void cuda_przynies(cudaStream_t stream){
		wartosci.cuda_przynies(stream);
		wartosci_zaabsorbowane.cuda_przynies(stream);
	}

	__host__ void cuda_free(){
		wartosci.cuda_free();
		wartosci_zaabsorbowane.cuda_free();
	}
};

};

__host__ void cuda_tester();
__HD__ void testy_macierzy();

// transformata ma miec pola: arrnosc, ile_watkow, i metode transformuj(index_watka)
template<typename towar, typename transformata>
struct spacer_losowy{
	// Normalnie wartosci sa przerzucane:
	//iteracja:  1   |  2   |  3   |  4   |  5
	//			A->B | B->A | A->B | B->A | A->B
	// I co niektóre iteracje mog¹ byæ zapamietane

	bool A = true; // true oznacza ze iteracjaA bedzie z a iteracjaB bedzie do
	spacer::dane_iteracji<towar> iteracjaA;
	spacer::dane_iteracji<towar> iteracjaB;

//               Pola zmienne na cudzie
//------------------------------------------
//				 Pola stale na cudzie

	wektor_do_pushbackowania<spacer::dane_iteracji<towar>*> iteracje_zapamietane; //iteracje_zapamietane[0] to stan poszatkowy
	spacer::dane_trwale<transformata> trwale;

	spacer_losowy* lokalizacja_na_device = nullptr;
	cudaStream_t stream = 0;

	__host__ spacer_losowy(const graf& przestrzen, uint32_t max_iteracji_zapamietanych = 30000)
	: trwale(przestrzen), iteracjaA(0), iteracjaB(0), iteracje_zapamietane(max_iteracji_zapamietanych){}

	__host__ spacer_losowy(const spacer_losowy& kopiowany)
	: trwale(kopiowany.trwale), iteracjaA(kopiowany.iteracjaA), iteracjaB(kopiowany.iteracjaB), A(kopiowany.A), iteracje_zapamietane(kopiowany.iteracje_zapamietane.rozmiar_zmallocowany){
		for(uint64_t i = 0; i < kopiowany.iteracje_zapamietane.rozmiar; i++){
			iteracje_zapamietane.pushback(new spacer::dane_iteracji<towar>(*(kopiowany.iteracje_zapamietane[i])));
		}
	}

	spacer_losowy& operator=(const spacer_losowy& kopiowany){
		trwale = kopiowany.trwale;
		iteracjaA = kopiowany.iteracjaA;
		iteracjaB = kopiowany.iteracjaB;
		A = kopiowany.A;
		
		// Opró¿nij zapamietane iteracje
		for (uint64_t i = 0; i < iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<towar>* ptr = iteracje_zapamietane[i];
			delete (ptr);
		}

		iteracje_zapamietane.przebuduj(kopiowany.iteracje_zapamietane.rozmiar_zmallocowany);
		for (uint64_t i = 0; i < kopiowany.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<towar>& temp = *(kopiowany.iteracje_zapamietane[i]);
			spacer::dane_iteracji<towar>* nowy = new spacer::dane_iteracji<towar>(temp);
			iteracje_zapamietane.pushback(nowy);
		}
		//this->~spacer_losowy();
		//std::construct_at<spacer_losowy>(this, kopiowany);
		return *this;
	}

	__host__ bool czy_gotowy(){
		bool ret = true;
		ret = trwale.czy_gotowy();
		// czy prawdopodobienstwo sumuje siê do 1.0;
		typ_prawdopodobienstwa prawdopodop = iteracjaA.prawdopodobienstwo_suma();
		if(abs(prawdopodop - 1.0) > dokladnosc){
			printf("Prawdopodobienstwo nie sumuje sie do 1.0; Brakuje %lf\n", abs(prawdopodop - 1.0));
			ret = false;
		}
		ASSERT_Z_ERROR_MSG(ret, "Spacer nie jest gotowy\n")
		return ret;
	}

	__host__ void przygotuj_pierwsza_iteracje(){
		iteracjaA = spacer::dane_iteracji<towar>(trwale.liczba_kubelkow(), trwale.liczba_absorberow());
		iteracjaA.zeruj();
		iteracjaA.czas = 0.0;

		iteracjaB = spacer::dane_iteracji<towar>(trwale.liczba_kubelkow(), trwale.liczba_absorberow());
	}

	__host__ void zapisz_iteracje(){		
		spacer::dane_iteracji<towar>* zapisywana = &iteracjaB;
		if(A) zapisywana = &iteracjaA;

		spacer::dane_iteracji<towar>* zapisana = new spacer::dane_iteracji<towar>(0);
		(*zapisana) = (*zapisywana);
		ASSERT_Z_ERROR_MSG((iteracje_zapamietane.rozmiar + 1)<=iteracje_zapamietane.rozmiar_zmallocowany , "Brakuje miejsca na kolejna iteracje\n");
		iteracje_zapamietane.pushback(zapisana);
	}

	__host__ void iteracja_na_cpu(){
		// nie musi korzystac ze znajdywacza wierzcholka w tak jak watki
		spacer::dane_iteracji<towar>* iteracja_z = &iteracjaA;
		spacer::dane_iteracji<towar>* iteracja_do = &iteracjaB;
		if(A == false){
			iteracja_z = &iteracjaB;
			iteracja_do = &iteracjaA;
		}

		for(uint64_t i = 0; i < trwale.wierzcholki.rozmiar; i++){
			spacer::wierzcholek& wierzcholek = trwale.wierzcholki[i];
			for (uint64_t j = 0; j < trwale.transformaty[wierzcholek.transformer].ile_watkow; j++) {
				trwale.transformaty[wierzcholek.transformer].transformuj(trwale, wierzcholek, *iteracja_z, *iteracja_do, j);
			}
		}
	}

	// najpierw iteracja
	__host__ void absorbuj_na_cpu(){
		spacer::dane_iteracji<towar>* iteracja_z = &iteracjaA;
		spacer::dane_iteracji<towar>* iteracja_do = &iteracjaB;
		if (A == false) {
			iteracja_z = &iteracjaB;
			iteracja_do = &iteracjaA;
		}

		for(uint64_t i = 0; i < trwale.liczba_absorberow(); i++){
			uint64_t indeks_absorbowany = trwale.indeksy_absorbowane[i];
			towar zaabsorbowane = iteracja_do->wartosci[indeks_absorbowany];
			iteracja_do->wartosci[indeks_absorbowany] = zero(towar());
			iteracja_do->wartosci_zaabsorbowane[i] = P(zaabsorbowane) + iteracja_z->wartosci_zaabsorbowane[i];
		}
	}

	__HD__ void dokoncz_iteracje(double delta_t){
		if(A){
			iteracjaB.czas = iteracjaA.czas + delta_t;
		} else {
			iteracjaA.czas = iteracjaB.czas + delta_t;
		}
		A = !A;
	}

	~spacer_losowy(){
		for(uint64_t i = 0; i < iteracje_zapamietane.rozmiar; i++){
			spacer::dane_iteracji<towar>* ptr = iteracje_zapamietane[i];
			delete (ptr);
		}
		if(lokalizacja_na_device != nullptr) zburz_na_cuda();
	}

	__host__ void zbuduj_na_cuda(){
		ASSERT_Z_ERROR_MSG(lokalizacja_na_device == nullptr, "Juz jest na cudzie\n");

		checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

		checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&lokalizacja_na_device), sizeof(spacer_losowy)));

		trwale.zbuduj_na_cuda(stream);

		iteracjaA.cuda_malloc();
		iteracjaA.cuda_zanies(stream);

		iteracjaB.cuda_malloc();
		iteracjaB.cuda_zanies(stream);

		checkCudaErrors(cudaMemcpyAsync((void*)lokalizacja_na_device, (void*)this, sizeof(spacer_losowy), cudaMemcpyHostToDevice, stream));

		checkCudaErrors(cudaStreamSynchronize(stream));
	}

	__host__ void zapisz_iteracje_z_cuda(){
		// Nie synchronizuje
		ASSERT_Z_ERROR_MSG(lokalizacja_na_device != nullptr, "Nic nie ma na urzadzeniu\n");
		spacer::dane_iteracji<towar>* zapisywana = &iteracjaB;
		if (A) zapisywana = &iteracjaA;

		spacer::dane_iteracji<towar>* zapisana = new spacer::dane_iteracji<towar>(trwale.liczba_kubelkow(), trwale.liczba_absorberow());
		
		zapisana->wartosci.pamiec_device = zapisywana->wartosci.pamiec_device;
		zapisana->wartosci.cuda_przynies(stream);
		zapisana->wartosci.pamiec_device = nullptr;

		zapisana->wartosci_zaabsorbowane.pamiec_device = zapisywana->wartosci_zaabsorbowane.pamiec_device;
		zapisana->wartosci_zaabsorbowane.cuda_przynies(stream);
		zapisana->wartosci_zaabsorbowane.pamiec_device = nullptr;

		checkCudaErrors(cudaMemcpyAsync(&(zapisana->czas),
			A ? &lokalizacja_na_device->iteracjaA.czas : &lokalizacja_na_device->iteracjaB.czas,
			sizeof(double), cudaMemcpyDeviceToHost, stream));

		ASSERT_Z_ERROR_MSG((iteracje_zapamietane.rozmiar + 1) <= iteracje_zapamietane.rozmiar_zmallocowany, "Brakuje miejsca na kolejna iteracje\n");
		iteracje_zapamietane.pushback(zapisana);
	}

	__host__ void cuda_przynies(){
		ASSERT_Z_ERROR_MSG(lokalizacja_na_device != nullptr, "Nie ma nic na cudzie\n");
		iteracjaA.cuda_przynies(stream);
		iteracjaB.cuda_przynies(stream);

		// g³ównie po to aby czasy iteracji by³y przepisane
		checkCudaErrors(cudaMemcpyAsync((void*)this, (void*)lokalizacja_na_device, offsetof(spacer_losowy, iteracje_zapamietane), cudaMemcpyDeviceToHost, stream));

		checkCudaErrors(cudaStreamSynchronize(stream));
	}

	__host__ void zburz_na_cuda(){
		ASSERT_Z_ERROR_MSG(lokalizacja_na_device != nullptr, "Nie ma nic na cudzie\n");

		trwale.zburz_na_cuda();

		iteracjaA.cuda_free();
		iteracjaB.cuda_free();

		checkCudaErrors(cudaFree(lokalizacja_na_device));
		lokalizacja_na_device = nullptr;

		checkCudaErrors(cudaStreamDestroy(stream));
		stream = 0;
	}

	void odwroc(){
		// dla dowolnego grafu to nie dzia³a, trzeba permutacje odwróciæ
		trwale.odwroc();
	}

	//__HD__ virtual void mala_praca(statyczny_wektor<towar>& docelowy, uint64_t index_watka){
	//	uint64_t index_wierzcholka = znajdz_wierzcholek(index_watka);
	//	uint64_t index_watka_w_wierzcholku = znajdywacz_wierzcholka[index_wierzcholka] - index_watka;
	//	spacer::wierzcholek& w = wierzcholki[index_wierzcholka];
	//	// tu jeszcze brakuje
	//}
};

__HD__ double dot(const estetyczny_wektor<double>&, const estetyczny_wektor<double>&);
__HD__ zesp dot(const estetyczny_wektor<zesp>&, const estetyczny_wektor<zesp>&);


//							---===SPACER LINIA===---
template<typename transformata>
__host__ spacer::uklad_transformat<transformata> uklad_transformat_dla_lini(uint32_t liczba_wierzcholkow, transformata& srodek, transformata& koniec);

template<typename towar, typename transformata>
__host__ spacer_losowy<towar, transformata> spacer_linia(
uint32_t liczba_wierzcholkow, transformata& srodek, transformata& konce, 
graf* linia = nullptr);


//							---===SPACER KRATA 2D===---
template<typename towar, typename transformata>
__host__ spacer_losowy<towar, transformata> spacer_krata_2D(
uint32_t liczba_wierzcholkow_boku, transformata srodek, transformata bok, transformata naroznik, 
graf* krata = nullptr);

template<typename transformata>
__host__ spacer::uklad_transformat<transformata> uklad_transformat_dla_kraty_2D(uint32_t liczba_wierzcholkow_boku, transformata& srodek, transformata& bok, transformata& naroznik);



template<typename towar, typename transformata>
__host__ void test_funkcji_tworzacych_spacery(transformata srodek, transformata konce);

template<typename towar, typename transformata>
__host__ void test_funkcji_tworzacych_spacery_2(transformata srodek, transformata bok, transformata narozniki);

//template __host__ spacer::uklad_transformat<transformata_macierz<double>> uklad_transformat_dla_lini<transformata_macierz<double>>(uint32_t liczba_wierzcholkow, transformata_macierz<double>& srodek, transformata_macierz<double>& koniec);

template <typename towar, typename transformata>
__global__ void iteracje_na_gpu(spacer_losowy<towar, transformata>* lokalizacja_na_device, uint64_t ile_blokow, uint64_t liczba_iteracji = 1, uint64_t ile_prac_wykonac = 1);


constexpr int max_ilosc_watkow_w_bloku = 100;

template <typename towar, typename transformata>
__host__ void iteruj_na_gpu(spacer_losowy<towar, transformata>& spacer,
	uint64_t liczba_iteracji = 1, uint64_t liczba_watkow = max_ilosc_watkow_w_bloku);

template <typename towar, typename transformata>
__host__ void iteracje_na_gpu(spacer_losowy<towar, transformata>& spacer,
	uint64_t liczba_iteracji, uint64_t ile_prac_na_watek, uint32_t ile_watkow_na_blok_max, uint32_t co_ile_zapisac);

