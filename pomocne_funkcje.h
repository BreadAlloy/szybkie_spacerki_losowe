#pragma once

#include "rzeczy_cudowe.h"

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

#define lepszy_assert(expression) (void)(                                                       \
            (!!(expression)) ||                                                              \
            (_wassert(_CRT_WIDE(#expression), _CRT_WIDE(__FILE__), (unsigned)(__LINE__)), 0) \
        )

#define SEP ,
#define ASSERT_Z_ERROR_MSG(warunek, wiadomosc) if(!(warunek)) {printf(wiadomosc); lepszy_assert(false); }

#define ASSERT_Z_INSTRUKCJA(warunek, instrukcja) if(!(warunek)) {instrukcja assert(false); }

//wiadomoœæ mo¿e byæ z formatem po przecinku tylko wtedy trzeba SEP u¿yæ na wyra¿eniu

static inline __HD__ double zero(double) {
	return 0.0;
}

static inline __HD__ double jeden(double) {
	return 1.0;
}
