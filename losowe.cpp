#include <random>
#include "zesp.h"
#include "losowe.h"

namespace losowosc_globalna {
	constexpr double dwa_pi = 3.1415926535897932 * 2.0;

	uint64_t nasionko = 1;

	std::uniform_real_distribution<double> losowe_przecinkowe(0.0, 1.0);
	std::uniform_real_distribution<double> losowe_kat(0.0, dwa_pi);
	std::mt19937_64 rng(nasionko);

	double __host__ losowa_przecinkowa() {
		return losowe_przecinkowe(rng);
	}

	double __host__ losowy_kat() {
		return losowe_przecinkowe(rng);
	}

	zesp __host__ losowy_zesp_naiwny() {
		return zesp(losowa_przecinkowa(), losowa_przecinkowa());
	}

	zesp __host__ losowy_zesp_z_okregu() {
		double kat = losowy_kat();
		return zesp(cos(kat), sin(kat));
	}

	zesp __host__ losowy_zesp_z_kola() {
		double kat = losowy_kat();
		return zesp(cos(kat), sin(kat)) * losowa_przecinkowa();
	}
};