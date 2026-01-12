#include "alg_liniowa.h"

bool __host__ ortonormalizuj(std::vector<zesp>& wektor, uint64_t arrnosc){
	ASSERT_Z_ERROR_MSG(wektor.size() == arrnosc*arrnosc, "Niepoprawny rozmiar wektora, albo arrnosc\n");
	for(uint64_t i = 0; i < arrnosc; i++){

		double norma = 0.0;
		for (uint64_t j = 0; j < arrnosc; j++) {
			norma += P(wektor[i * arrnosc + j]);
		}
		if (norma < 10e-10) {
			return false; // s³aba ortonormalizacja
		}

		double normalizator = 1.0 / NORMA(1.0, norma, zesp());
		for (uint64_t j = 0; j < arrnosc; j++) {
			wektor[i * arrnosc + j] *= normalizator;
		}


		for(uint64_t j = i + 1UL; j < arrnosc; j++){
			uint64_t indeks_staly = i*arrnosc;
			uint64_t indeks_zmieniany = j*arrnosc;
			estetyczny_wektor<zesp> staly(&(wektor[indeks_staly]), arrnosc);
			estetyczny_wektor<zesp> zmieniany(&(wektor[indeks_zmieniany]), arrnosc);

			zesp przykrycie = dot(staly, zmieniany);

			for(uint64_t k = 0; k < arrnosc; k++){
				zmieniany[k] = zmieniany[k] - przykrycie * staly[k];
			}
		}
	}
	
	//for (uint64_t i = 0; i < arrnosc; i++) {
	//	double norma = 0.0;
	//	for (uint64_t j = 0; j < arrnosc; j++) {
	//		norma += P(wektor[i * arrnosc + j]);
	//	}
	//	if(norma < 10e-6){
	//		return false; // s³aba ortonormalizacja
	//	}

	//	double normalizator = 1.0/NORMA(1.0, norma, zesp());
	//	for (uint64_t j = 0; j < arrnosc; j++) {
	//		wektor[i * arrnosc + j] *= normalizator;
	//	}
	//}

	return true; // dobra ortonormalizacja
}

zesp __HD__ expi(double x){
	return zesp(cos(x), sin(x));
}

transformata_macierz<zesp> __host__ transformata_postac_ogolna(double theta, double alpha, double beta, double gamma){
	zesp faza_globalna = expi(alpha);
	return transformata_macierz<zesp>(faza_globalna * expi(beta) * cos(theta), faza_globalna * expi(gamma) * sin(theta),
									-faza_globalna * expi(-gamma) * sin(theta), faza_globalna * expi(-beta) * cos(theta));
}

transformata_macierz<zesp> __host__ losowa_transformata(uint64_t arrnosc){
	std::vector<zesp> wektor(arrnosc * arrnosc);
	while(true){
		for(auto& z : wektor){
			z = losowosc_globalna::losowy_zesp_z_kola();
		}
		if(ortonormalizuj(wektor, arrnosc)){
			break;
		}
	}
	return transformata_macierz<zesp>(arrnosc, wektor.data());
}

void __host__ test_ortonormalizacji(){
	losowa_transformata(1);
	losowa_transformata(2);
	losowa_transformata(4);
	losowa_transformata(10);
}


