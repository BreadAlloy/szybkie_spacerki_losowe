#pragma once

typedef double fp_typ;

constexpr inline __HD__ fp_typ zero(fp_typ) {
	return ((fp_typ)0.0);
}

constexpr inline __HD__ fp_typ jeden(fp_typ) {
	return ((fp_typ)1.0);
}

namespace config{

#define NORMALIZACJA false
constexpr bool normalizacja = NORMALIZACJA;

#define ABSORBCJA false
constexpr bool absorbcja = ABSORBCJA;

#define MIERZENIE_ROZPROSZONE true
constexpr bool mierzenie_rozproszone = MIERZENIE_ROZPROSZONE;

}

