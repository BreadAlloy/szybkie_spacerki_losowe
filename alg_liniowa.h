#pragma once
#include <vector>

#include "transformaty.h"
#include "losowe.h"
#include "wektory.h"
#include "spacer_losowy.h"

bool __host__ ortonormalizuj(std::vector<zesp>& wektor, uint64_t arrnosc);

zesp __HD__ expi(double x);

transformata_macierz<zesp> __host__ transformata_postac_ogolna(double theta, double alpha, double beta, double gamma);

transformata_macierz<zesp> __host__ losowa_transformata(uint64_t arrnosc);

void __host__ test_ortonormalizacji();


