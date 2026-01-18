#pragma once

#include "spacer_losowy.h"

#include "transformaty.h"

#include "zesp.h"

struct transformata_macierz_dyskretna_klasyczna : transformata_macierz<double>{
	typedef transformata_macierz_dyskretna_klasyczna TMDK;

	transformata_macierz_dyskretna_klasyczna(transformata_macierz<double> M)
	: transformata_macierz<double>(M){}

	__HD__ void transformuj(spacer::dane_trwale<TMDK>& trwale, const spacer::wierzcholek& wierzcholek,
		spacer::dane_iteracji<double>& iteracja_z, spacer::dane_iteracji<double>& iteracja_do, uint64_t index_w_wierzcholku, uint64_t)
	{
		TMDK& transformata = trwale.transformaty[wierzcholek.transformer];
		estetyczny_wektor<double> a(&(iteracja_z[wierzcholek.start_wartosci]), wierzcholek.liczba_kierunkow);
		estetyczny_wektor<double> b(&(transformata((uint8_t)index_w_wierzcholku, 0)), transformata.arrnosc);
		iteracja_do[(uint64_t)trwale.gdzie_wyslac[wierzcholek.start_wartosci + index_w_wierzcholku]] = dot(b, a);
	}
};

typedef transformata_macierz_dyskretna_klasyczna TMDK;

struct transformata_macierz_dyskretna_kwantowa : transformata_macierz<zesp> {
	typedef transformata_macierz_dyskretna_kwantowa TMDQ;

	transformata_macierz_dyskretna_kwantowa(transformata_macierz<zesp> M)
		: transformata_macierz<zesp>(M) {}

	__HD__ void transformuj(spacer::dane_trwale<TMDQ>& trwale, const spacer::wierzcholek& wierzcholek,
		spacer::dane_iteracji<zesp>& iteracja_z, spacer::dane_iteracji<zesp>& iteracja_do, uint64_t index_w_wierzcholku, uint64_t)
	{
		TMDQ& transformata = trwale.transformaty[wierzcholek.transformer];
		estetyczny_wektor<zesp> a(&(iteracja_z[wierzcholek.start_wartosci]), wierzcholek.liczba_kierunkow);
		estetyczny_wektor<zesp> b(&(transformata((uint8_t)index_w_wierzcholku, 0)), transformata.arrnosc);
		iteracja_do[(uint64_t)trwale.gdzie_wyslac[wierzcholek.start_wartosci + index_w_wierzcholku]] = dot(b, a);
	}
};

typedef transformata_macierz_dyskretna_kwantowa TMDQ;


constexpr double dt = 0.01;

struct transformata_macierz_ciagla_kwantowa : transformata_macierz<zesp> {
	typedef transformata_macierz_ciagla_kwantowa TMCQ;

	// const
	zesp schrodinger = zesp(0.0, -1.0) * dt;

	transformata_macierz_ciagla_kwantowa(transformata_macierz<zesp> M)
		: transformata_macierz<zesp>(M) {
		//ile_watkow = 1;
	}

	__HD__ void transformuj(spacer::dane_trwale<TMCQ>& trwale, const spacer::wierzcholek& wierzcholek,
		spacer::dane_iteracji<zesp>& iteracja_z, spacer::dane_iteracji<zesp>& iteracja_do, uint64_t index_w_wierzcholku, uint64_t)
	{
		estetyczny_wektor<zesp> a(&(iteracja_z[wierzcholek.start_wartosci]), wierzcholek.liczba_kierunkow);
		estetyczny_wektor<zesp> b(&(operator()((uint8_t)index_w_wierzcholku, 0)), arrnosc);
		uint64_t offset_do = (uint64_t)trwale.gdzie_wyslac[wierzcholek.start_wartosci + index_w_wierzcholku];
		zesp PSI = (iteracja_z[offset_do] + schrodinger * dot(b, a)) * iteracja_z.norma_poprzedniej_iteracji;
		iteracja_do[offset_do] = PSI;
	}

	__HD__ void transformuj_rozniczka(spacer::dane_trwale<TMCQ>* trwale, const spacer::wierzcholek& wierzcholek,
		zesp* zrodlo, zesp* zwrot, uint64_t index_w_wierzcholku, uint64_t)
	{
		estetyczny_wektor<zesp> a(&(zrodlo[wierzcholek.start_wartosci]), wierzcholek.liczba_kierunkow);
		estetyczny_wektor<zesp> b(&(operator()((uint8_t)index_w_wierzcholku, 0)), arrnosc);
		uint64_t offset_do = (uint64_t)trwale->gdzie_wyslac[wierzcholek.start_wartosci + index_w_wierzcholku];
		zwrot[offset_do] = dot(b, a);
	}
};

typedef transformata_macierz_ciagla_kwantowa TMCQ;