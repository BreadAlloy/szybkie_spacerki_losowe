#pragma once
#include <vector>

#include "transformaty.h"
#include "losowe.h"
#include "wektory.h"
#include "spacer_losowy.h"

bool __host__ ortonormalizuj(std::vector<zesp>& wektor, uint64_t arrnosc);

zesp __HD__ expi(fp_t x);

transformata_macierz<zesp> __host__ transformata_postac_ogolna(fp_t theta, fp_t alpha, fp_t beta, fp_t gamma);

transformata_macierz<zesp> __host__ losowa_transformata(uint64_t arrnosc);

void __host__ test_ortonormalizacji();


