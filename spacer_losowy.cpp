#include "spacer_losowy.h"

#include "transformaty_wyspecializowane.h"


//										 ---===GRAF LINIA===---
template<typename towar, typename transformata>
__host__ spacer_losowy<towar, transformata> spacer_linia(
uint32_t liczba_wierzcholkow, transformata T, transformata boki, graf* linia) {
	bool zkasuj = false;
	if(linia == nullptr){
		//czy to nie jest niewydajne?
		linia = new graf(graf_lini(liczba_wierzcholkow, BEZ_NAZW));
		zkasuj = true;
	}

	spacer_losowy<towar, transformata> spacer(*linia);
	spacer::uklad_transformat<transformata> transformaty = uklad_transformat_dla_lini<transformata>(liczba_wierzcholkow, T, boki);
	spacer.trwale.dodaj_transformaty(transformaty);
	spacer.trwale.przygotuj_znajdywacz_wierzcholka();
	spacer.przygotuj_pierwsza_iteracje();
	spacer.iteracjaA[spacer.trwale.wierzcholki[liczba_wierzcholkow / 2].start_wartosci] = jeden(towar());
	spacer.czy_gotowy();

	if(zkasuj) delete linia;
	return spacer;
}

template __host__ spacer_losowy<double, TMDK> spacer_linia(
	uint32_t liczba_wierzcholkow, TMDK T, TMDK boki, graf* linia);

template __host__ spacer_losowy<zesp, TMDQ> spacer_linia(
	uint32_t liczba_wierzcholkow, TMDQ T, TMDQ boki, graf* linia);

template __host__ spacer_losowy<zesp, TMCQ> spacer_linia(
	uint32_t liczba_wierzcholkow, TMCQ T, TMCQ boki, graf* linia);

template<typename transformata>
__host__ spacer::uklad_transformat<transformata> uklad_transformat_dla_lini(uint32_t liczba_wierzcholkow, transformata& T, transformata& koniec) {
	spacer::uklad_transformat<transformata> uklad(liczba_wierzcholkow);

	uint64_t id_T = uklad.dodaj_transformate(T);
	for (uint64_t i = 1; i < ((uint64_t)liczba_wierzcholkow - 1UL); i++) {
		uklad.podepnij_transformate(id_T, i);
	}

	uint64_t id_koniec = uklad.dodaj_transformate(koniec);
	uklad.podepnij_transformate(id_koniec, 0);
	uklad.podepnij_transformate(id_koniec, liczba_wierzcholkow - 1);

	return uklad;
}

//											---===GRAF KRATA 2D===---
template<typename towar, typename transformata>
__host__ spacer_losowy<towar, transformata> spacer_krata_2D(
uint32_t liczba_wierzcholkow_boku, transformata T, transformata boki, 
graf* krata) {
	bool zkasuj = false;
	if (krata == nullptr) {
		//czy to nie jest niewydajne?
		krata = new graf(graf_krata_2D(liczba_wierzcholkow_boku, BEZ_NAZW));
		zkasuj = true;
	}

	spacer_losowy<towar, transformata> spacer(*krata);
	spacer::uklad_transformat<transformata> transformaty = uklad_transformat_dla_kraty_2D<transformata>(liczba_wierzcholkow_boku, T, boki);
	spacer.trwale.dodaj_transformaty(transformaty);
	spacer.trwale.przygotuj_znajdywacz_wierzcholka();
	spacer.przygotuj_pierwsza_iteracje();
	spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(towar());
	spacer.czy_gotowy();

	if (zkasuj) delete krata;
	return spacer;
}

template __host__ spacer_losowy<double, TMDK> spacer_krata_2D(
	uint32_t liczba_wierzcholkow_boku, TMDK T, TMDK boki,
	graf* krata);

template __host__ spacer_losowy<zesp, TMDQ> spacer_krata_2D(
	uint32_t liczba_wierzcholkow_boku, TMDQ T, TMDQ boki,
	graf* krata);

template __host__ spacer_losowy<zesp, TMCQ> spacer_krata_2D(
	uint32_t liczba_wierzcholkow_boku, TMCQ T, TMCQ boki,
	graf* krata);

template<typename transformata>
__host__ spacer::uklad_transformat<transformata> uklad_transformat_dla_kraty_2D(uint32_t liczba_wierzcholkow_boku, transformata& T, transformata& boki) {
	uint32_t liczba_wierzcholkow = liczba_wierzcholkow_boku * liczba_wierzcholkow_boku;
	spacer::uklad_transformat<transformata> uklad(liczba_wierzcholkow);

	uint64_t id_T = uklad.dodaj_transformate(T);
	for (uint64_t i = 1; i < (liczba_wierzcholkow_boku - (uint32_t)1); i++) {
		for (uint64_t j = 1; j < (liczba_wierzcholkow_boku - (uint32_t)1); j++) {
			uklad.podepnij_transformate(id_T, i*liczba_wierzcholkow_boku + j);
		}
	}

	uint64_t id_boki = uklad.dodaj_transformate(boki);
	for (uint64_t i = 1; i < (liczba_wierzcholkow_boku - (uint32_t)1); i++) {
		uklad.podepnij_transformate(id_boki, i * liczba_wierzcholkow_boku);
		uklad.podepnij_transformate(id_boki, i * liczba_wierzcholkow_boku + liczba_wierzcholkow_boku - 1);
	}

	for (uint64_t i = 0; i < (uint64_t)liczba_wierzcholkow_boku; i++) {
		uklad.podepnij_transformate(id_boki, i);
		uklad.podepnij_transformate(id_boki, liczba_wierzcholkow - 1 - i);
	}

	return uklad;
}

template<typename towar, typename transformata>
__host__ spacer_losowy<towar, transformata> spacer_krata_2D_cykl(
	uint32_t liczba_wierzcholkow_boku, transformata T, graf* krata) {
	bool zkasuj = false;
	if (krata == nullptr) {
		//czy to nie jest niewydajne?
		krata = new graf(graf_krata_2D_cykl(liczba_wierzcholkow_boku, BEZ_NAZW));
		zkasuj = true;
	}

	spacer_losowy<towar, transformata> spacer(*krata);
	spacer::uklad_transformat<transformata> transformaty = uklad_transformat_dla_kraty_2D<transformata>(liczba_wierzcholkow_boku, T, T);
	spacer.trwale.dodaj_transformaty(transformaty);
	spacer.trwale.przygotuj_znajdywacz_wierzcholka();
	spacer.przygotuj_pierwsza_iteracje();
	spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(towar());
	spacer.czy_gotowy();

	if (zkasuj) delete krata;
	return spacer;
}

template __host__ spacer_losowy<zesp, TMCQ> spacer_krata_2D_cykl(
	uint32_t liczba_wierzcholkow_boku, TMCQ T, graf* krata);

template<typename towar, typename transformata>
__host__ void test_funkcji_tworzacych_spacery(transformata T, transformata bok) {
	spacer_losowy<towar, transformata> spacer_linia3 = spacer_linia<towar, transformata>(3, T, bok);
	spacer_losowy<towar, transformata> spacer_linia10 = spacer_linia<towar, transformata>(10, T, bok);
	spacer_losowy<towar, transformata> spacer_linia1000 = spacer_linia<towar, transformata>(1000, T, bok);

	for (uint64_t i = 0; i < 10; i++) {
		spacer_linia1000.iteracja_na_cpu();
		spacer_linia1000.dokoncz_iteracje(1.0);
	}
}

template<typename towar, typename transformata>
__host__ void test_funkcji_tworzacych_spacery_2(transformata T, transformata bok) {
	spacer_losowy<towar, transformata> spacer_krata_2D_3 = spacer_krata_2D<towar, transformata>(3, T, bok);
	spacer_losowy<towar, transformata> spacer_krata_2D_10 = spacer_krata_2D<towar, transformata>(10, T, bok);
	spacer_losowy<towar, transformata> spacer_krata_2D_1000 = spacer_krata_2D<towar, transformata>(1000, T, bok);
	for (uint64_t i = 0; i < 10; i++) {
		spacer_krata_2D_1000.iteracja_na_cpu();
		spacer_krata_2D_1000.dokoncz_iteracje(1.0);
		spacer_krata_2D_1000.zapisz_iteracje();
	}
}

template __host__ void test_funkcji_tworzacych_spacery_2<double, TMDK>(TMDK T, TMDK bok);

template __host__ void test_funkcji_tworzacych_spacery<double, TMDK>(TMDK T, TMDK bok);

//void test4(){}

//template __host__ spacer::uklad_transformat<transformata_macierz<double>> uklad_transformat_dla_lini<transformata_macierz<double>>(uint32_t liczba_wierzcholkow, transformata_macierz<double>& srodek, transformata_macierz<double>& koniec);
//template __host__ spacer::uklad_transformat<transformata> uklad_transformat_dla_lini<transformata_macierz<double>>(uint32_t liczba_wierzcholkow, transformata& srodek, transformata& koniec);

