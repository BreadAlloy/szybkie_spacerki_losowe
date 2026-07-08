#pragma once

__HD__ __forceinline__ fp_t dot(const estetyczny_wektor<fp_t>& a, const estetyczny_wektor<fp_t>& b) {
	ASSERT_Z_ERROR_MSG(a.rozmiar == b.rozmiar, "Dot product na innej ilosci elementow\n");
	fp_t sum = FP_ZERO;
	for (uint64_t i = 0; i < a.rozmiar; i++) {
		sum += (a[i] * b[i]);
	}
	return sum;
}

__HD__ __forceinline__ zesp dot(const estetyczny_wektor<zesp>& a, const estetyczny_wektor<zesp>& b) {
	ASSERT_Z_ERROR_MSG(a.rozmiar == b.rozmiar, "Dot product na innej ilosci elementow\n");
	zesp sum = ZESP_ZERO;
	for (uint64_t i = 0; i < a.rozmiar; i++) {
		sum += (a[i].sprzezenie() * b[i]);
	}
	return sum;
}

__HD__ __forceinline__ fp_t dot(const fp_t* a, const fp_t* b, uint8_t ilosc) {
	fp_t sum = FP_ZERO;
	for (uint8_t i = 0; i < ilosc; i++) {
		sum += (a[i] * b[i]);
	}
	return sum;
}

__HD__ __forceinline__ zesp dot(const zesp* a, const zesp* b, uint8_t ilosc) {
	zesp sum = ZESP_ZERO;
	for (uint8_t i = 0; i < ilosc; i++) {
		sum += (a[i].sprzezenie() * b[i]);
	}
	return sum;
}