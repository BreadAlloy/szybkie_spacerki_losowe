#pragma once
	
#include "rzeczy_cudowe.h"

#include "spacer_config.h"

constexpr inline __HD__ fp_t zero(fp_t) {
	return ((fp_t)0.0);
}

#undef FP_ZERO
#define FP_ZERO zero(fp_t())

constexpr inline __HD__ fp_t jeden(fp_t) {
	return ((fp_t)1.0);
}

#define FP_JEDEN jeden(fp_t())

#ifdef NDEBUG
	#undef NDEBUG
	#include <cassert>  // asserty w trybie z debugiem
	#define NDEBUG 1
#else
	#include <cassert>  // asserty tez w trybie z debugiem
#endif

#undef assert

_ACRTIMP void __cdecl _wassert(
    _In_z_ wchar_t const* _Message,
    _In_z_ wchar_t const* _File,
    _In_   unsigned       _Line
);

#define spacer_assert(expression) (void)(                                                       \
            (!!(expression)) ||                                                              \
            (_wassert(_CRT_WIDE(#expression), _CRT_WIDE(__FILE__), (unsigned)(__LINE__)), 0) \
        )

#define SEP ,
#define ASSERT_Z_ERROR_MSG(warunek, wiadomosc) if(!(warunek)) {printf(wiadomosc); spacer_assert(false); }

#define ASSERT_Z_INSTRUKCJA(warunek, instrukcja) if(!(warunek)) {instrukcja spacer_assert(false); }

#define NIE_ZAIMPLEMENTOWANE_ERROR(wiadomosc) ASSERT_Z_ERROR_MSG(false, "Nie zaimplementowane: "##wiadomosc)

//wiadomoœæ mo¿e byæ z formatem po przecinku tylko wtedy trzeba SEP u¿yæ na wyra¿eniu

#include <chrono>

#define CZAS_INIT std::chrono::steady_clock::time_point begin;\
				  std::chrono::steady_clock::time_point end;\
				  long int diff;

#define CZAS_START begin = std::chrono::steady_clock::now();

#define CZAS_STOP end = std::chrono::steady_clock::now();\
				  diff = (long int)std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count();\
				  printf("Trwalo: %d ms\n", diff/1000);

#include <type_traits>
#include <limits>
