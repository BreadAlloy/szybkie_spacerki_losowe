#pragma once

#include "imgui.h"

#include "transformaty.h"

#include "spacer_losowy.h"

template<typename towar>
__host__ void pokaz_transformate(transformata_macierz<towar>& op);

template<typename towar>
__host__ void pokaz_stan(const estetyczny_wektor<towar>& wartosci);


