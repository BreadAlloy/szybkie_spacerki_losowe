#pragma once

#include "graf.h"

#include "wektory.h"

#include "transformaty.h"

#include "zesp.h"

typedef double typ_prawdopodobienstwa;

static inline double P(double a){
	return a;
}

static inline double P(const zesp& a) {
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

template<typename transformata>
struct dane_trwale{ //operatory, to gdzie wysy³aæ, przestrzen, raczej nie zmienia sie z czasem
	statyczny_wektor<uint64_t> gdzie_wyslac; // krawedzie w grafie
	statyczny_wektor<spacer::wierzcholek> wierzcholki;
	statyczny_wektor<uint64_t> znajdywacz_wierzcholka; // znajduje wierzcholek na podstawie indexu watka
	statyczny_wektor<transformata> transformaty;

	__host__ dane_trwale(const graf& przestrzen) {
		ASSERT_Z_ERROR_MSG(przestrzen.czy_gotowy(), "graf nie byl gotowy\n");

		wierzcholki.malloc(przestrzen.liczba_wierzcholkow());
		znajdywacz_wierzcholka.malloc(przestrzen.liczba_wierzcholkow());

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

	__host__ void przygotuj_znajdywacz_wierzcholka(){
		uint64_t znajdywacz_wierzcholka_iterator = 0;
		for (ID_W i = 0; i < wierzcholki.rozmiar; i++) {
			znajdywacz_wierzcholka_iterator += transformaty[(wierzcholki[i]).transformer].ile_watkow;
			znajdywacz_wierzcholka[i] = znajdywacz_wierzcholka_iterator;
		}
	}

	__host__ uint64_t ile_watkow(){
		return znajdywacz_wierzcholka[znajdywacz_wierzcholka.rozmiar - 1];
	}

	__host__ uint64_t liczba_kubelkow(){
		return gdzie_wyslac.rozmiar;
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

	__host__ void przeslij_na_cuda(dane_trwale* na_cudzie, cudaStream_t stream){
		//przenies_na_cuda(na_cudzie->gdzie_wyslac, gdzie_wyslac);
		//przenies_na_cuda(na_cudzie->wierzcholki, wierzcholki);
		//przenies_na_cuda(na_cudzie->znajdywacz_wierzcholka, znajdywacz_wierzcholka);
		//przenies_na_cuda(na_cudzie->transformaty, transformaty);

		//checkCudaErrors(cudaMemcpyAsync((void*)lokalizacja_na_cuda, (void*)na_cudzie, sizeof(dane_trwale), cudaMemcpyHostToDevice, stream));
	}

	__HD__ uint64_t liczba_wierzcholkow(){
		return wierzcholki.rozmiar;
	}

	__HD__ uint64_t znajdz_wierzcholek(uint64_t index_watka) {
		// zwraca index szukanego wierzcho³ka
		for (uint64_t i = 0; i < znajdywacz_wierzcholka.rozmiar; i++) {
			if (index_watka < znajdywacz_wierzcholka[i]) return i;
		}
		ASSERT_Z_ERROR_MSG(false, "Nie znaleziono wierzcholka?!?!?!?\n");
		return (uint64_t)-1;
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
};

template<typename towar>
struct dane_iteracji{
	statyczny_wektor<towar> wartosci;
	double czas = 0.0;

	dane_iteracji(uint64_t liczba_wartosci = 0)
	: wartosci(liczba_wartosci) {}

	void zeruj(){
		wartosci.memset(zero(towar()));
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
};

};

__host__ void cuda_tester();
__HD__ void testy_macierzy();

// transformata ma miec pola: arrnosc, ile_watkow, i metode transformuj(index_watka)
template<typename towar, typename transformata>
struct spacer_losowy{
	spacer::dane_trwale<transformata> trwale;

	// Normalnie wartosci sa przerzucane:
	//iteracja:  1   |  2   |  3   |  4   |  5
	//			A->B | B->A | A->B | B->A | A->B
	// I co niektóre iteracje mog¹ byæ zapamietane

	bool A = true; // true oznacza ze iteracjaA bedzie z a iteracjaB bedzie do
	spacer::dane_iteracji<towar> iteracjaA;
	spacer::dane_iteracji<towar> iteracjaB;
	wektor_do_pusbackowania<spacer::dane_iteracji<towar>*> iteracje_zapamietane; //iteracje_zapamietane[0] to stan poszatkowy

	//zmienna_miedzy_HD<spacer_losowy> translator;

	__host__ spacer_losowy(const graf& przestrzen, uint32_t max_iteracji_zapamietanych = 30000)
	: trwale(przestrzen), iteracjaA(0), iteracjaB(0), iteracje_zapamietane(max_iteracji_zapamietanych){}

	__host__ spacer_losowy(const spacer_losowy& kopiowany)
	: trwale(kopiowany.trwale), iteracjaA(kopiowany.iteracjaA), iteracjaB(kopiowany.iteracjaB), A(kopiowany.A), iteracje_zapamietane(kopiowany.iteracje_zapamietane.rozmiar_zmallocowany){
		for(uint64_t i = 0; i < kopiowany.iteracje_zapamietane.rozmiar; i++){
			iteracje_zapamietane.pushback(new spacer::dane_iteracji<towar>(*(kopiowany.iteracje_zapamietane[i])));
		}
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
		iteracjaA = spacer::dane_iteracji<towar>(trwale.liczba_kubelkow());
		iteracjaA.zeruj();
		iteracjaA.czas = 0.0;

		iteracjaB = spacer::dane_iteracji<towar>(trwale.liczba_kubelkow());
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

	__host__ void dokoncz_iteracje(double dt){
		if(A){
			iteracjaB.czas = iteracjaA.czas + dt;
		} else {
			iteracjaA.czas = iteracjaB.czas + dt;
		}
		A = !A;
	}

	~spacer_losowy(){
		for(uint64_t i = 0; i < iteracje_zapamietane.rozmiar; i++){
			delete (iteracje_zapamietane[i]);
		}
	}

	__host__ void wyslij_na_cuda(){
		//ASSERT_Z_ERROR_MSG(lokalizacja_na_cuda == nullptr, "Juz jest na cudzie");
		//checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(lokalizacja_na_cuda), sizeof(spacer_losowy)));

		spacer_losowy na_cudzie;
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

