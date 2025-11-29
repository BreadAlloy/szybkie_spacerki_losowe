#pragma once

#include "spacer_losowy.h"

#include "transformaty.h"

#include "zesp.h"

struct transformata_macierz_dyskretna_klasyczna : transformata_macierz<double>{
	typedef transformata_macierz_dyskretna_klasyczna TMDK;

	transformata_macierz_dyskretna_klasyczna(transformata_macierz<double> M)
	: transformata_macierz<double>(M){}

	__HD__ void transformuj(spacer::dane_trwale<TMDK>& trwale, const spacer::wierzcholek& wierzcholek,
		spacer::dane_iteracji<double>& iteracja_z, spacer::dane_iteracji<double>& iteracja_do, uint64_t index_w_wierzcholku)
	{
		TMDK& transformata = trwale.transformaty[wierzcholek.transformer];
		estetyczny_wektor<double> a(&(iteracja_z[wierzcholek.start_wartosci]), wierzcholek.liczba_kierunkow);
		estetyczny_wektor<double> b(&(transformata((uint8_t)index_w_wierzcholku, 0)), transformata.arrnosc);
		iteracja_do[trwale.gdzie_wyslac[wierzcholek.start_wartosci + index_w_wierzcholku]] = dot(b, a);
	}
};

typedef transformata_macierz_dyskretna_klasyczna TMDK;

struct transformata_macierz_dyskretna_kwantowa : transformata_macierz<zesp> {
	typedef transformata_macierz_dyskretna_kwantowa TMDQ;

	transformata_macierz_dyskretna_kwantowa(transformata_macierz<zesp> M)
		: transformata_macierz<zesp>(M) {}

	__HD__ void transformuj(spacer::dane_trwale<TMDQ>& trwale, const spacer::wierzcholek& wierzcholek,
		spacer::dane_iteracji<zesp>& iteracja_z, spacer::dane_iteracji<zesp>& iteracja_do, uint64_t index_w_wierzcholku)
	{
		TMDQ& transformata = trwale.transformaty[wierzcholek.transformer];
		estetyczny_wektor<zesp> a(&(iteracja_z[wierzcholek.start_wartosci]), wierzcholek.liczba_kierunkow);
		estetyczny_wektor<zesp> b(&(transformata((uint8_t)index_w_wierzcholku, 0)), transformata.arrnosc);
		iteracja_do[trwale.gdzie_wyslac[wierzcholek.start_wartosci + index_w_wierzcholku]] = dot(b, a);
	}
};

typedef transformata_macierz_dyskretna_kwantowa TMDQ;