#pragma once

#include "wektory.h"

#include "pomocne_funkcje.h"

#include "zesp.h"

__host__ std::string do_bin(const double&);

__host__ std::string do_bin(const zesp&);

template<typename towar>
struct transformata_macierz {
	uint8_t arrnosc = 0;
	uint8_t ile_watkow = 0;
	statyczny_wektor<towar> macierz;
	/*
		---=== macierz ===---
			 [[0, 1, 2],
			  [3, 4, 5],
			  [6, 7, 8]]
	*/

	__HD__ transformata_macierz(const transformata_macierz& kopiowana) :
		arrnosc(kopiowana.arrnosc), ile_watkow(kopiowana.ile_watkow), macierz(kopiowana.macierz)
	{
	}

	__HD__ transformata_macierz<towar>& operator=(const transformata_macierz<towar>& kopiowana) {
		this->~transformata_macierz();
		this->transformata_macierz::transformata_macierz(kopiowana);
		sprawdz();
		return *this;
	}

	__HD__ transformata_macierz(towar a00)
		: arrnosc(1), ile_watkow(1) {
		malloc_na_arrnosc();
		(*this)(0, 0) = a00;
		sprawdz();
	}

	__HD__ transformata_macierz(towar a00, towar a01,
		towar a10, towar a11)
		: arrnosc(2), ile_watkow(2) {
		malloc_na_arrnosc();
		(*this)(0, 0) = a00;
		(*this)(1, 0) = a10;
		(*this)(0, 1) = a01;
		(*this)(1, 1) = a11;
		sprawdz();
	}

	__HD__ transformata_macierz(uint8_t arrnosc)
		: arrnosc(arrnosc), ile_watkow(arrnosc) {
		malloc_na_arrnosc();
		macierz.memset(zero(towar()));
	}

	__HD__ transformata_macierz(uint8_t arrnosc, towar* zawartosc) :
		transformata_macierz(arrnosc) {
		void* res = memcpy(macierz.pamiec, zawartosc, sizeof(towar) * ile_elementow());
		ASSERT_Z_ERROR_MSG(this->macierz.pamiec == res, "malloc nie zadzialal\n");
		sprawdz();
	}

	__HD__ ~transformata_macierz() {
		macierz.~statyczny_wektor();
	}

	__HD__ void malloc_na_arrnosc() {
		macierz.malloc(ile_elementow());
	}

	__HD__ uint32_t ile_elementow() {
		return (uint32_t)arrnosc * (uint32_t)arrnosc;
	}

	__host__ std::string str() const { // przy okazji od razu jest to numpy macierz
		std::string result = "[";
		for (uint8_t i = 0; i < arrnosc; i++) {
			result += "[";
			for (uint8_t j = 0; j < arrnosc; j++) {
				result += do_bin((*this)(i, j));
				if (j != arrnosc - 1) result += ", ";
			}
			if (i != (arrnosc - 1)) {
				result += "],\n";
			}
			else {
				result += "]]";
			}
		}
		return result;
	}

	__HD__ bool sprawdz() const {
		bool ret = sprawdz_poprawnosc(*this);
		ASSERT_Z_ERROR_MSG(ret, "Macierz nie jest poprawna\n");
		return ret;
	}

	__HD__ bool operator==(const transformata_macierz& b) const {
		const transformata_macierz& a = *this;
		if (a.arrnosc != b.arrnosc) return false;
		for (uint8_t i = 0; i < a.arrnosc; i++) {
			for (uint8_t j = 0; j < a.arrnosc; j++) {
				if (a(i, j) != b(i, j)) return false;
			}
		}
		return true;
	}

	__HD__ towar& operator()(uint8_t row, uint8_t column) {
		lepszy_assert(row < arrnosc);
		lepszy_assert(column < arrnosc);
		return macierz[(uint32_t)row * (uint32_t)arrnosc + (uint32_t)column];
	}

	__HD__ towar operator()(uint8_t row, uint8_t column) const {
		lepszy_assert(row < arrnosc);
		lepszy_assert(column < arrnosc);
		return macierz[(uint32_t)row * (uint32_t)arrnosc + (uint32_t)column];
	}
};


template<typename towar>
__host__ void pokaz_transformate(transformata_macierz<towar>& op);

template<typename towar>
__host__ void pokaz_stan(const estetyczny_wektor<towar>& wartosci);

template<typename towar>
__HD__ transformata_macierz<towar> tensor(const transformata_macierz<towar>&, const transformata_macierz<towar>&);

template<typename towar>
__HD__ transformata_macierz<towar> mnoz(const transformata_macierz<towar>&, const transformata_macierz<towar>&);

__HD__ transformata_macierz<zesp> hermituj(const transformata_macierz<zesp>&);

__HD__ bool sprawdz_poprawnosc(const transformata_macierz<zesp>&);

__HD__ bool sprawdz_poprawnosc(const transformata_macierz<double>&);
