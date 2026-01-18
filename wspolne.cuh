#pragma once

__HD__ __forceinline__ double dot(const estetyczny_wektor<double>& a, const estetyczny_wektor<double>& b) {
	ASSERT_Z_ERROR_MSG(a.rozmiar == b.rozmiar, "Dot product na innej ilosci elementow\n");
	double sum = 0.0;
	for (uint64_t i = 0; i < a.rozmiar; i++) {
		sum += (a[i] * b[i]);
	}
	return sum;
}

__HD__ __forceinline__ zesp dot(const estetyczny_wektor<zesp>& a, const estetyczny_wektor<zesp>& b) {
	ASSERT_Z_ERROR_MSG(a.rozmiar == b.rozmiar, "Dot product na innej ilosci elementow\n");
	zesp sum = zesp(0.0, 0.0);
	for (uint64_t i = 0; i < a.rozmiar; i++) {
		sum += (a[i].sprzezenie() * b[i]);
	}
	return sum;
}