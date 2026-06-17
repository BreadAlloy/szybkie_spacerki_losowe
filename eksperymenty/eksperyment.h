#pragma once
#include "przejrzenie_reczne.h"

namespace eksperyment {

	struct zderzenie{
		const uint32_t liczba_wierzcholkow_boku = 201;		

		okno_przegladania<TMCQ>* okno = nullptr;
		graf przestrzen;
		spacer_losowy<zesp, TMCQ> spacer;

		zderzenie(TMCQ& transformer)
		: przestrzen(graf_krata_2D_cykl(liczba_wierzcholkow_boku))
		, spacer(spacer_krata_2D_cykl<zesp, TMCQ>(liczba_wierzcholkow_boku, transformer, &przestrzen)) 
		{

			spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = zero(zesp());
			spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(zesp()) / sqrt(4.0);
			spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci + 2] = jeden(zesp()) / sqrt(4.0);

			spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 4 - 14].start_wartosci + 1] = jeden(zesp()) / sqrt(4.0);
			spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 4 - 14].start_wartosci + 3] = jeden(zesp()) / sqrt(4.0);

			okno = new okno_przegladania<TMCQ>(
				"Eksperyment: zderzenie", 4.0f,
				liczba_wierzcholkow_boku, &przestrzen, spacer);
		}

		void pokaz_okno(){
			okno->pokaz_okno();
		}

		~zderzenie(){
			delete okno;
		}

	};











};

