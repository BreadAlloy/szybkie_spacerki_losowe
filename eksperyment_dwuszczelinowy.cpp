#include "eksperyment_dwuszczelinowy.h"

template<typename towar, typename transformata>
__host__ spacer_losowy<towar, transformata> spacer_eksperymentu_dwuszczelinowego(
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
	spacer::indeksy_pozycji pozycje_absorberow;
	for(uint64_t i = 0; i < liczba_wierzcholkow_boku; i++){
		pozycje_absorberow.push_back(spacer::indeks_pozycji(liczba_wierzcholkow_boku * i + (liczba_wierzcholkow_boku - 1) - 5, 0));
	}
	//for (uint64_t i = 0; i < liczba_wierzcholkow_boku; i++) {
	//	pozycje_absorberow.push_back(spacer::indeks_pozycji(liczba_wierzcholkow_boku * i + 5, 0));
	//}
	//for (uint64_t i = 1; i < liczba_wierzcholkow_boku - 1; i++) {
	//	pozycje_absorberow.push_back(spacer::indeks_pozycji(i, 2));
	//	pozycje_absorberow.push_back(spacer::indeks_pozycji(liczba_wierzcholkow_boku*(liczba_wierzcholkow_boku - 1) + i, 2));
	//}
	spacer.trwale.dodaj_absorbery(pozycje_absorberow);
	spacer.przygotuj_pierwsza_iteracje();
	spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(towar());
	spacer.czy_gotowy();

	if (zkasuj) delete krata;
	return spacer;
}

template __host__ spacer_losowy<zesp, TMDQ> spacer_eksperymentu_dwuszczelinowego(
	uint32_t liczba_wierzcholkow_boku, TMDQ srodek, TMDQ bok, TMDQ naroznik,
	graf* krata);

template __host__ spacer_losowy<zesp, TMCQ> spacer_eksperymentu_dwuszczelinowego(
	uint32_t liczba_wierzcholkow_boku, TMCQ srodek, TMCQ bok, TMCQ naroznik,
	graf* krata);
