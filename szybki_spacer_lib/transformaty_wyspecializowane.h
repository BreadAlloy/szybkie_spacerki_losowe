#pragma once

#include "spacer_losowy.h"

#include "transformaty.h"

#include "zesp.h"

struct transformata_macierz_dyskretna_klasyczna : transformata_macierz<fp_t>{
	typedef transformata_macierz_dyskretna_klasyczna TMDK;

	transformata_macierz_dyskretna_klasyczna(transformata_macierz<fp_t> M)
	: transformata_macierz<fp_t>(M){}

	__HD__ void transformuj(spacer::dane_trwale<TMDK>& trwale, const spacer::wierzcholek& wierzcholek,
		spacer::dane_iteracji<fp_t>& iteracja_z, spacer::dane_iteracji<fp_t>& iteracja_do, uint64_t index_w_wierzcholku, uint64_t)
	{
		TMDK& transformata = trwale.transformaty[wierzcholek.transformer];
		estetyczny_wektor<fp_t> a(&(iteracja_z[wierzcholek.start_wartosci]), wierzcholek.liczba_kierunkow);
		estetyczny_wektor<fp_t> b(&(transformata((uint8_t)index_w_wierzcholku, 0)), transformata.arrnosc);
		iteracja_do[(uint64_t)trwale.gdzie_wyslac[wierzcholek.start_wartosci + index_w_wierzcholku]] = dot(b, a);
	}
};

typedef transformata_macierz_dyskretna_klasyczna TMDK;

struct transformata_macierz_dyskretna_kwantowa : transformata_macierz<zesp> {
	typedef transformata_macierz_dyskretna_kwantowa TMDQ;

	transformata_macierz_dyskretna_kwantowa(transformata_macierz<zesp> M)
		: transformata_macierz<zesp>(M) {}

	__HD__ void transformuj_proste(const zesp* przed, zesp* po, uint8_t index_w_wierzcholku) {
		zesp* wiersz_macierzy = &(this->operator()(index_w_wierzcholku, 0));
		zesp suma = dot(wiersz_macierzy, przed, arrnosc);
		po[index_w_wierzcholku] = suma;
	}

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


constexpr fp_t dt = fp_t(0.1);

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
		ASSERT_Z_ERROR_MSG(config::normalizacja, "Tego w tym configu nie wolamy\n");
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

struct transformata_macierz_schrodingerowata_kwantowa : transformata_macierz<zesp> {
	typedef transformata_macierz_schrodingerowata_kwantowa TMSQ;

	// const
	zesp schrodinger = zesp(0.0, -1.0) * dt;

	transformata_macierz_schrodingerowata_kwantowa(transformata_macierz<zesp> M)
		: transformata_macierz<zesp>(M) {
		ile_watkow = 1;
	}

	__HD__ void transformuj_proste(const zesp* przed, zesp* po, uint8_t index_w_wierzcholku) {
		ASSERT_Z_ERROR_MSG(index_w_wierzcholku == 0, "Mial byc 1 watek\n");

		for(uint8_t k = 0; k < arrnosc; k++){
			zesp* wiersz_macierzy = &(this->operator()(k, 0));
			zesp suma = dot(wiersz_macierzy, przed, arrnosc);
			po[k] = przed[k] + suma * schrodinger;
		}

		prob_t suma_P_przed = FP_ZERO;
		for (uint8_t k = 0; k < arrnosc; k++) {
			suma_P_przed += P(przed[k]);
		}

		prob_t suma_P_po = FP_ZERO;
		for (uint8_t k = 0; k < arrnosc; k++) {
			suma_P_po += P(po[k]);
		}

		prob_t norma = NORMA_stabilna(suma_P_po, suma_P_przed, zesp());
		for(uint8_t k = 0; k < arrnosc; k++){
			po[k] *= norma;
		}
		
	}
};

typedef transformata_macierz_schrodingerowata_kwantowa TMSQ;