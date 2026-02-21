#include "rozniczka.cuh"
#include "wspolne.cuh"

__host__ podobienstwo_liniowe dopasuj_liniowo(statyczny_wektor<zesp>& A, statyczny_wektor<zesp>& B) {
	//szuka A*x + roznica == B 
	ASSERT_Z_ERROR_MSG(A.rozmiar == B.rozmiar, "Rozne rozmiary do dopasowania\n");

	podobienstwo_liniowe podob = podobienstwo_liniowe(A.rozmiar);

	double rozmiar = (double)podob.blad.rozmiar;
	double suma_wag = 0.0;
	zesp suma_skalarow = zesp(0.0, 0.0);
	for (uint64_t i = 0; i < podob.blad.rozmiar; i++) {
		if (A[i].norm() < 1e-8 && B[i].norm() < 1e-8) continue;
		zesp lokalny_skalar = (B[i] / A[i]);
		double waga_skalaru = A[i].norm();
		suma_skalarow += lokalny_skalar * waga_skalaru;
		suma_wag += waga_skalaru;
		//if ((lokalny_skalar * A[i] - B[i]).norm() < 1e-4) {
		//}
		//else {
		//	rozmiar--;
		//}
	}
	podob.x = suma_skalarow / suma_wag;
	for (uint64_t i = 0; i < podob.blad.rozmiar; i++) {
		podob.blad[i] = A[i] * podob.x - B[i];
	}
	return podob;
}

__host__ podobienstwo_liniowe dopasuj_liniowo_norma(statyczny_wektor<zesp>& A, statyczny_wektor<zesp>& B) {
	//szuka  - A*x + roznica == B 
	ASSERT_Z_ERROR_MSG(A.rozmiar == B.rozmiar, "Rozne rozmiary do dopasowania\n");

	podobienstwo_liniowe podob = podobienstwo_liniowe(A.rozmiar);

	double suma_prawdopodob_A = 0.0;
	double suma_prawdopodob_B = 0.0;
	for (uint64_t i = 0; i < podob.blad.rozmiar; i++) {
		suma_prawdopodob_A += P(A[i]);
		suma_prawdopodob_B += P(B[i]);
	}
	podob.x = zesp(1.0, 0.0) * NORMA(suma_prawdopodob_A, suma_prawdopodob_B, zesp());
	for (uint64_t i = 0; i < podob.blad.rozmiar; i++) {
		podob.blad[i] = podob.x * A[i] + B[i];
	}
	return podob;
}