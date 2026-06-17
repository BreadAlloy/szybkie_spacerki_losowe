#include <random>
#include "zesp.h"
#include "losowe.h"

namespace losowosc_globalna {
	constexpr fp_t dwa_pi = fp_t(3.1415926535897932 * 2.0);

	uint64_t nasionko = 1;

	std::uniform_real_distribution<fp_t> losowe_przecinkowe(FP_ZERO, FP_JEDEN);
	std::uniform_real_distribution<fp_t> losowe_kat(FP_ZERO, dwa_pi);
	std::mt19937_64 rng(nasionko);

	fp_t __host__ losowa_przecinkowa() {
		return losowe_przecinkowe(rng);
	}

	fp_t __host__ losowy_kat() {
		return losowe_przecinkowe(rng);
	}

	zesp __host__ losowy_zesp_naiwny() {
		return zesp(losowa_przecinkowa(), losowa_przecinkowa());
	}

	zesp __host__ losowy_zesp_z_okregu() {
		fp_t kat = losowy_kat();
		return zesp(cos(kat), sin(kat));
	}

	zesp __host__ losowy_zesp_z_kola() {
		fp_t kat = losowy_kat();
		return zesp(cos(kat), sin(kat)) * losowa_przecinkowa();
	}
};