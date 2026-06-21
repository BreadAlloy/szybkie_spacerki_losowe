#include <gtest/gtest.h>

#include "spacer_losowy.h"

#include "transformaty_wyspecializowane.h"

#include "definicje_typowych_macierzy.h"

#include "okno_benchmarku.h"

// ---=== CONFIG ===---

typedef zesp towar_benchowany;
typedef TMDQ transformata_benchowana;
#define TEMPLATY_BENCHOWANE <towar_benchowany, transformata_benchowana>
typedef spacer_losowy TEMPLATY_BENCHOWANE spacer_benchowany;

constexpr uint32_t liczba_iteracji = 2000;

constexpr uint64_t ile_prac_na_watek = 70;
constexpr uint32_t ile_watkow_na_blok_max = 300;

constexpr uint32_t ile_benchow_na_lini = 11;
constexpr uint64_t benchowane_rozmiary_lini[] =
{500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 2000000, 4000000, 10000000};

constexpr uint32_t ile_benchow_na_kracie_2d = 10;
constexpr uint32_t benchowane_rozmiary_kraty_2d[] =
{ 22, 32, 71, 100, 223, 316, 707, 1000, 1400, 2000, 3162};

constexpr uint32_t ile_benchow_liczb_czastek = 6;
constexpr uint32_t benchowane_liczby_czastek[] =
{ 1, 2, 3, 4, 5, 6, 7};
constexpr uint32_t rozmiar_bazowy = 4;

std::vector<rezultat_benchu_2> rezultaty_benchu;

void iteruj_spacer(spacer_benchowany spacer, rezultat_benchu_2& rezultat, 
    rezultat_benchu_2& min_pamiec, rezultat_benchu_2& min_flops) {

    auto [aprox_min_pamiec, aprox_min_flops] = spacer.theoretical_performance(liczba_iteracji);
    min_pamiec.zaloguj(float(spacer.trwale.liczba_kubelkow()), float(aprox_min_pamiec));
    min_flops.zaloguj(float(spacer.trwale.liczba_kubelkow()), float(aprox_min_flops));

    CZAS_INIT

        spacer.zbuduj_na_cuda();

    CZAS_START
        proste_iteracje_na_gpu(
            spacer, FP_JEDEN, liczba_iteracji,
            ile_prac_na_watek, ile_watkow_na_blok_max,
            liczba_iteracji + 1);
    CZAS_STOP

        rezultat.zaloguj(float(spacer.trwale.liczba_kubelkow()), float(diff / 1000));

    spacer.zburz_na_cuda();

}

TEST(RozmiarSpaceru, Linia) {
    rezultat_benchu_2 rezultat("Linia");
    rezultat_benchu_2 min_pamiec("min-pamiec Linia");
    rezultat_benchu_2 min_flops("min-flops Linia");

    for (uint32_t i = 0; i < ile_benchow_na_lini; i++) {
            uint32_t rozmiar = benchowane_rozmiary_lini[i];

            spacer_benchowany spacer = spacer_linia TEMPLATY_BENCHOWANE
            (rozmiar, H, H);
            iteruj_spacer(spacer, rezultat, min_pamiec, min_flops);
    }

    rezultaty_benchu.push_back(rezultat);
    rezultaty_benchu.push_back(min_pamiec);
    rezultaty_benchu.push_back(min_flops);
}

TEST(RozmiarSpaceru, Grid2d) {
    rezultat_benchu_2 rezultat("Grid2d");
    rezultat_benchu_2 min_pamiec("min-pamiec Grid2d");
    rezultat_benchu_2 min_flops("min-flops Grid2d");

    for (uint32_t i = 0; i < ile_benchow_na_kracie_2d; i++) {
        uint32_t rozmiar = benchowane_rozmiary_kraty_2d[i];

        spacer_benchowany spacer = spacer_krata_2D TEMPLATY_BENCHOWANE
        (rozmiar, Fourier_4, Fourier_4);
        iteruj_spacer(spacer, rezultat, min_pamiec, min_flops);
    }

    rezultaty_benchu.push_back(rezultat);
    rezultaty_benchu.push_back(min_pamiec);
    rezultaty_benchu.push_back(min_flops);
}

TEST(RozmiarSpaceru, Nczastek) {
    rezultat_benchu_2 rezultat("Linia N czastek");
    rezultat_benchu_2 min_pamiec("min-pamiec Linia N czastek");
    rezultat_benchu_2 min_flops("min-flops Linia N czastek");

    graf przestrzen = graf_lini(rozmiar_bazowy, BEZ_NAZW);
    for (uint32_t i = 0; i < ile_benchow_liczb_czastek; i++) {
        uint32_t liczba_czastek = benchowane_liczby_czastek[i];
        
        transformata_benchowana T = H;
        for(uint32_t j = 1; j < liczba_czastek; j++){
            T = tensor(T, H);
        }

        graf przestrzen_wieksza = przestrzen.tensorowy(liczba_czastek);
        spacer_benchowany spacer(przestrzen_wieksza);
        spacer::uklad_transformat<transformata_benchowana> uklad =
            uklad_transformat_wszystko_to_samo(
                spacer.trwale.liczba_wierzcholkow(), T);
        spacer.trwale.dodaj_transformaty(uklad);
        spacer.trwale.przygotuj_znajdywacz_wierzcholka();
        spacer.przygotuj_pierwsza_iteracje();
        spacer.iteracjaA[0] = jeden(towar_benchowany());
        spacer.czy_gotowy();

        iteruj_spacer(spacer, rezultat, min_pamiec, min_flops);
    }

    rezultaty_benchu.push_back(rezultat);
    rezultaty_benchu.push_back(min_pamiec);
    rezultaty_benchu.push_back(min_flops);
}

TEST(Koniec, PokazRezultat) {
    okno_benchmarku okno("Rozmiar grafu");

    while (okno.tick_start()) {

        if (ImGui::Begin("Rezultaty")) {

            if (ImPlot::BeginPlot("##Rezultaty", ImVec2(800.0f, 800.0f))) {
                ImPlot::SetupAxes("Rozmiar", "czas[ms]");

                for (auto& rezultaty : rezultaty_benchu) {
                    rezultaty.pokaz_dane();
                }
                ImPlot::EndPlot();
            }
            ImGui::End();
        }

        okno.tick_finish();
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

/*
1. liczba iteracji
2. rozmiar lini
3. rozmiar kraty
4. kilka czastek na lini
5. parametry cudy | liczba prac na watek, max liczba watkow w bloku
6. czestotliwoœæ zapisywania

1 - typ spaceru, spacer, liczby iteracji
2 - typ spaceru, rozmiary lini, liczba iteracji
3 - typ spaceru, rozmiary kraty, liczba iteracji
4 - typ spaceru, liczby czastek, liczba iteracji
5 - typ spaceru, spacer, liczby prac, liczby watkow na blok, liczba iteracji
*/



