#pragma once

typedef float fp_t;

typedef fp_t prob_t;

constexpr fp_t fp_tolerancja = (fp_t)1.0e-6;

#define FP_F "f"

#define FP_POS_F ".9f" 

namespace config{

#define NORMALIZACJA false
constexpr bool normalizacja = NORMALIZACJA;

#define ABSORBCJA false
constexpr bool absorbcja = ABSORBCJA;

#define MIERZENIE_ROZPROSZONE true
constexpr bool mierzenie_rozproszone = MIERZENIE_ROZPROSZONE;

}

typedef uint32_t ID_W; // typ indeksowania wierzcholków
typedef uint8_t ID_K;  // typ indeksowania kube³ków

typedef ID_K arrnosc_t;

