#pragma once

#include "spacer_losowy.h"

#include "transformaty_wyspecializowane.h"

template<typename towar, typename transformata>
__host__ spacer_losowy<towar, transformata> spacer_eksperymentu_dwuszczelinowego(
	uint32_t liczba_wierzcholkow_boku, transformata srodek, transformata bok, transformata naroznik,
	graf* krata = nullptr);
