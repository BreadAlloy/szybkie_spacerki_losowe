#pragma once
#include <random>
#include "zesp.h"

namespace losowosc_globalna {
	extern uint64_t nasionko;

	extern std::uniform_real_distribution<double> losowe_przecinkowe;
	extern std::uniform_real_distribution<double> losowe_kat;
	extern std::mt19937_64 rng;

	double __host__ losowa_przecinkowa();

	double __host__ losowy_kat();

	zesp __host__ losowy_zesp_naiwny();

	zesp __host__ losowy_zesp_z_okregu();

	zesp __host__ losowy_zesp_z_kola();
};