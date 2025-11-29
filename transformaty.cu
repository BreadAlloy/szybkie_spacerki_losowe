#pragma once

#include "transformaty.h"

#include "spacer_losowy.h" // tylko po to by tolerancje zna³o

template<typename towar>
__HD__ transformata_macierz<towar> tensor(const transformata_macierz<towar>& a, const transformata_macierz<towar>& b) {
	transformata_macierz<towar> result((uint8_t)(a.arrnosc * b.arrnosc));
	for (uint8_t i = 0; i < a.arrnosc; i++) {
		for (uint8_t j = 0; j < a.arrnosc; j++) {
			for (uint8_t k = 0; k < b.arrnosc; k++) {
				for (uint8_t l = 0; l < b.arrnosc; l++) {
					result(b.arrnosc * i + k, b.arrnosc * j + l) = a(i, j) * b(k, l);
				}
			}
		}
	}
	return result;
}

template __HD__ transformata_macierz<double> tensor(const transformata_macierz<double>& a, const transformata_macierz<double>& b);
template __HD__ transformata_macierz<zesp> tensor(const transformata_macierz<zesp>& a, const transformata_macierz<zesp>& b);

template<typename towar>
__HD__ transformata_macierz<towar> mnoz(const transformata_macierz<towar>& a, const transformata_macierz<towar>& b) {
	ASSERT_Z_ERROR_MSG(a.arrnosc == b.arrnosc, "Macierze maja inna arrnosc\n");
	transformata_macierz<towar> result(a);
	for (uint8_t i = 0; i < a.arrnosc; i++) {
		for (uint8_t j = 0; j < a.arrnosc; j++) {
			towar temp = zero(towar());
			for (uint8_t k = 0; k < a.arrnosc; k++) {
				temp += (a(i, k)) * (b(k, j));
			}
			result(i, j) = temp;
		}
	}
	return result;
}

template __HD__ transformata_macierz<double> mnoz(const transformata_macierz<double>& a, const transformata_macierz<double>& b);
template __HD__ transformata_macierz<zesp> mnoz(const transformata_macierz<zesp>&a, const transformata_macierz<zesp>&b);


__HD__ transformata_macierz<zesp> hermituj(const transformata_macierz<zesp>& x) {
	transformata_macierz<zesp> result(x);
	for (uint8_t i = 0; i < x.arrnosc; i++) {
		for (uint8_t j = 0; j < x.arrnosc; j++) {
			result(i, j) = x(j, i).sprzezenie();
		}
	}
	return result;
}

__HD__ bool sprawdz_poprawnosc(const transformata_macierz<zesp>& x) {
	transformata_macierz<zesp> zhermitowane = hermituj(x);
	transformata_macierz<zesp> temp = mnoz(x, hermituj(x));
	bool ret = true;
	for (uint8_t i = 0; i < temp.arrnosc; i++) {
		for (uint8_t j = 0; j < temp.arrnosc; j++) {
			if (abs(temp(i, j).Im) > tolerancja){
					ret = false;
					IF_HOST(printf("Na pozycji (%d, %d), w Im nie jest zero: %lf\n", i, j, temp(i, j).Im);)
				}
			if (i == j) {
				if (abs(temp(i, j).Re - 1.0) > tolerancja){
					ret = false;
					IF_HOST(printf("Na pozycji (%d, %d), w Im nie jest jeden: %lf\n", i, j, temp(i, j).Re);)
				}
			}
			else {
				if (abs(temp(i, j).Re) > tolerancja){
					ret = false;
					IF_HOST(printf("Na pozycji (%d, %d), w Re nie jest zero: %lf\n", i, j, temp(i, j).Re);)
				}
			}
		}
	}
	if (!ret) {
		IF_HOST(printf("Macierz odbiega od unitarnisci: %s\n", x.str().c_str()));
	}
	return ret;
}

__HD__ bool sprawdz_poprawnosc(const transformata_macierz<double>& x) {
	// sprawdzenie poprawnosci macierzy transformaty klasycznie
	bool ret = true;
	for (uint8_t k = 0; k < x.arrnosc; k++) {
		double suma = 0.0;
		for (uint8_t w = 0; w < x.arrnosc; w++) {
			suma += x(w, k);
		}
		if (abs(suma - 1.0) > tolerancja) {
			ret = false;
			IF_HOST(printf("Kolumna %d: odbiega od poprawnosci o: %.17lf\n", k, suma - 1.0);)
		}
	}
	if (!ret) {
		IF_HOST(
			printf("W macierzy:\n");
		printf("%s\n", x.str().c_str());
			)
	}
	return ret;
}

#ifdef __CUDA_ARCH__
#include "zesp.cu"
#endif
