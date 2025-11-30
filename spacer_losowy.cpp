#include "spacer_losowy.h"

#include "transformaty_wyspecializowane.h"


//										 ---===GRAF LINIA===---
template<typename towar, typename transformata>
__host__ spacer_losowy<towar, transformata> spacer_linia(
uint32_t liczba_wierzcholkow, transformata& srodek, transformata& konce, graf* linia) {
	bool zkasuj = false;
	if(linia == nullptr){
		//czy to nie jest niewydajne?
		linia = new graf(graf_lini(liczba_wierzcholkow, BEZ_NAZW));
		zkasuj = true;
	}

	spacer_losowy<towar, transformata> spacer(*linia);
	spacer::uklad_transformat<transformata> transformaty = uklad_transformat_dla_lini<transformata>(liczba_wierzcholkow, srodek, konce);
	spacer.trwale.dodaj_transformaty(transformaty);
	spacer.trwale.przygotuj_znajdywacz_wierzcholka();
	spacer.przygotuj_pierwsza_iteracje();
	spacer.iteracjaA[spacer.trwale.wierzcholki[liczba_wierzcholkow / 2].start_wartosci] = jeden(towar());
	spacer.czy_gotowy();

	if(zkasuj) delete linia;
	return spacer;
}

template __host__ spacer_losowy<double, TMDK> spacer_linia(
	uint32_t liczba_wierzcholkow, TMDK& srodek, TMDK& konce, graf* linia);

template __host__ spacer_losowy<zesp, TMDQ> spacer_linia(
	uint32_t liczba_wierzcholkow, TMDQ& srodek, TMDQ& konce, graf* linia);

template __host__ spacer_losowy<zesp, TMCQ> spacer_linia(
	uint32_t liczba_wierzcholkow, TMCQ& srodek, TMCQ& konce, graf* linia);

template<typename transformata>
__host__ spacer::uklad_transformat<transformata> uklad_transformat_dla_lini(uint32_t liczba_wierzcholkow, transformata& srodek, transformata& koniec) {
	spacer::uklad_transformat<transformata> uklad(liczba_wierzcholkow);
	uint64_t id_koniec = uklad.dodaj_transformate(koniec);
	uklad.podepnij_transformate(id_koniec, 0);
	uklad.podepnij_transformate(id_koniec, liczba_wierzcholkow - 1);

	uint64_t id_srodek = uklad.dodaj_transformate(srodek);
	for (uint64_t i = 1; i < (liczba_wierzcholkow - (uint32_t)1); i++) {
		uklad.podepnij_transformate(id_srodek, i);
	}
	return uklad;
}

//											---===GRAF KRATA 2D===---
template<typename towar, typename transformata>
__host__ spacer_losowy<towar, transformata> spacer_krata_2D(
uint32_t liczba_wierzcholkow_boku, transformata srodek, transformata bok, transformata naroznik, 
graf* krata) {
	bool zkasuj = false;
	if (krata == nullptr) {
		//czy to nie jest niewydajne?
		krata = new graf(graf_krata_2D(liczba_wierzcholkow_boku, BEZ_NAZW));
		zkasuj = true;
	}

	spacer_losowy<towar, transformata> spacer(*krata);
	spacer::uklad_transformat<transformata> transformaty = uklad_transformat_dla_kraty_2D<transformata>(liczba_wierzcholkow_boku, srodek, bok, naroznik);
	spacer.trwale.dodaj_transformaty(transformaty);
	spacer.trwale.przygotuj_znajdywacz_wierzcholka();
	spacer.przygotuj_pierwsza_iteracje();
	spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(towar());
	spacer.czy_gotowy();

	if (zkasuj) delete krata;
	return spacer;
}

template __host__ spacer_losowy<double, TMDK> spacer_krata_2D(
	uint32_t liczba_wierzcholkow_boku, TMDK srodek, TMDK bok, TMDK naroznik,
	graf* krata);

template __host__ spacer_losowy<zesp, TMDQ> spacer_krata_2D(
	uint32_t liczba_wierzcholkow_boku, TMDQ srodek, TMDQ bok, TMDQ naroznik,
	graf* krata);

template __host__ spacer_losowy<zesp, TMCQ> spacer_krata_2D(
	uint32_t liczba_wierzcholkow_boku, TMCQ srodek, TMCQ bok, TMCQ naroznik,
	graf* krata);

template<typename transformata>
__host__ spacer::uklad_transformat<transformata> uklad_transformat_dla_kraty_2D(uint32_t liczba_wierzcholkow_boku, transformata& srodek, transformata& bok, transformata& naroznik) {
	uint32_t liczba_wierzcholkow = liczba_wierzcholkow_boku * liczba_wierzcholkow_boku;
	spacer::uklad_transformat<transformata> uklad(liczba_wierzcholkow);

	uint64_t id_srodek = uklad.dodaj_transformate(srodek);
	for (uint64_t i = 1; i < (liczba_wierzcholkow_boku - (uint32_t)1); i++) {
		for (uint64_t j = 1; j < (liczba_wierzcholkow_boku - (uint32_t)1); j++) {
			uklad.podepnij_transformate(id_srodek, i*liczba_wierzcholkow_boku + j);
		}
	}

	uint64_t id_bok = uklad.dodaj_transformate(bok);
	for (uint64_t i = 1; i < (liczba_wierzcholkow_boku - (uint32_t)1); i++) {
		uklad.podepnij_transformate(id_bok, i * liczba_wierzcholkow_boku);
		uklad.podepnij_transformate(id_bok, i * liczba_wierzcholkow_boku + liczba_wierzcholkow_boku - 1);
	}

	for (uint64_t i = 1; i < (liczba_wierzcholkow_boku - (uint32_t)1); i++) {
		uklad.podepnij_transformate(id_bok, i);
		uklad.podepnij_transformate(id_bok, liczba_wierzcholkow - 1 - i);
	}

	uint64_t id_naroznik = uklad.dodaj_transformate(naroznik);
	uklad.podepnij_transformate(id_naroznik, 0);
	uklad.podepnij_transformate(id_naroznik, liczba_wierzcholkow_boku - 1);
	uklad.podepnij_transformate(id_naroznik, liczba_wierzcholkow - 1);
	uklad.podepnij_transformate(id_naroznik, liczba_wierzcholkow - liczba_wierzcholkow_boku);

	return uklad;
}



template<typename towar, typename transformata>
__host__ void test_funkcji_tworzacych_spacery(transformata srodek, transformata konce) {
	spacer_losowy<towar, transformata> spacer_linia3 = spacer_linia<towar, transformata>(3, srodek, konce);
	spacer_losowy<towar, transformata> spacer_linia10 = spacer_linia<towar, transformata>(10, srodek, konce);
	spacer_losowy<towar, transformata> spacer_linia1000 = spacer_linia<towar, transformata>(1000, srodek, konce);

	for (uint64_t i = 0; i < 10; i++) {
		spacer_linia1000.iteracja_na_cpu();
		spacer_linia1000.dokoncz_iteracje(1.0);
	}
}

template<typename towar, typename transformata>
__host__ void test_funkcji_tworzacych_spacery_2(transformata srodek, transformata bok, transformata narozniki) {
	spacer_losowy<towar, transformata> spacer_krata_2D_3 = spacer_krata_2D<towar, transformata>(3, srodek, bok, narozniki);
	spacer_losowy<towar, transformata> spacer_krata_2D_10 = spacer_krata_2D<towar, transformata>(10, srodek, bok, narozniki);
	spacer_losowy<towar, transformata> spacer_krata_2D_1000 = spacer_krata_2D<towar, transformata>(1000, srodek, bok, narozniki);
	for (uint64_t i = 0; i < 10; i++) {
		spacer_krata_2D_1000.iteracja_na_cpu();
		spacer_krata_2D_1000.dokoncz_iteracje(1.0);
		spacer_krata_2D_1000.zapisz_iteracje();
	}
}

template __host__ void test_funkcji_tworzacych_spacery_2<double, TMDK>(TMDK srodek, TMDK bok, TMDK narozniki);

template __host__ void test_funkcji_tworzacych_spacery<double, TMDK>(TMDK srodek, TMDK konce);

//void test4(){}

//template __host__ spacer::uklad_transformat<transformata_macierz<double>> uklad_transformat_dla_lini<transformata_macierz<double>>(uint32_t liczba_wierzcholkow, transformata_macierz<double>& srodek, transformata_macierz<double>& koniec);
//template __host__ spacer::uklad_transformat<transformata> uklad_transformat_dla_lini<transformata_macierz<double>>(uint32_t liczba_wierzcholkow, transformata& srodek, transformata& koniec);

