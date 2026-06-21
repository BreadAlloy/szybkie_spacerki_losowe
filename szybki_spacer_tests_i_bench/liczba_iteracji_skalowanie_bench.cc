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

constexpr uint32_t liczba_instancji = 6;
constexpr uint32_t benchowane_liczby_iteracji[] = 
    {0, 500, 1000, 2000, 5000, 10000};

constexpr uint64_t ile_prac_na_watek = 70;
constexpr uint32_t ile_watkow_na_blok_max = 300;

std::vector<rezultat_benchu_2> rezultaty_benchu;

void iteruj_spacer(spacer_benchowany spacer, 
        rezultat_benchu_2& rezultat, rezultat_benchu_2& min_pamiec, 
        rezultat_benchu_2& min_flops, uint32_t liczba_iteracji){
    printf("liczba iteracji: %d\n", liczba_iteracji);

    auto [aprox_min_pamiec, aprox_min_flops] = spacer.theoretical_performance(liczba_iteracji);
    min_pamiec.zaloguj(float(liczba_iteracji), float(aprox_min_pamiec));
    min_flops.zaloguj(float(liczba_iteracji), float(aprox_min_flops));

    CZAS_INIT

    spacer.zbuduj_na_cuda();

    CZAS_START
        proste_iteracje_na_gpu(
            spacer, FP_JEDEN, liczba_iteracji,
            ile_prac_na_watek, ile_watkow_na_blok_max,
            liczba_iteracji + 1);
    CZAS_STOP

    rezultat.zaloguj(float(liczba_iteracji), float(diff / 1000));

    spacer.zburz_na_cuda();

}

TEST(SkalowanieLiczbyInstancji, Linia) {
    spacer_benchowany spacer = spacer_linia TEMPLATY_BENCHOWANE
        (2000000, H, H);

    rezultat_benchu_2 rezultat("Linia");
    rezultat_benchu_2 min_pamiec("min-pamiec Linia");
    rezultat_benchu_2 min_flops("min-flops Linia");

    for(uint32_t i = 0; i < liczba_instancji; i++){
        uint32_t liczba_iteracji = benchowane_liczby_iteracji[i];
        iteruj_spacer(spacer, rezultat, min_pamiec, min_flops, liczba_iteracji);
    }

    rezultaty_benchu.push_back(rezultat);
    rezultaty_benchu.push_back(min_pamiec);
    rezultaty_benchu.push_back(min_flops);
}

TEST(SkalowanieLiczbyInstancji, Grid2d) {
    spacer_benchowany spacer = spacer_krata_2D TEMPLATY_BENCHOWANE
    (1000, Fourier_4, Fourier_4);

    rezultat_benchu_2 rezultat("Grid2d");
    rezultat_benchu_2 min_pamiec("min-pamiec Grid2d");
    rezultat_benchu_2 min_flops("min-flops Grid2d");

    for (uint32_t i = 0; i < liczba_instancji; i++) {
        uint32_t liczba_iteracji = benchowane_liczby_iteracji[i];
        iteruj_spacer(spacer, rezultat, min_pamiec, min_flops, liczba_iteracji);
    }

    rezultaty_benchu.push_back(rezultat);
    rezultaty_benchu.push_back(min_pamiec);
    rezultaty_benchu.push_back(min_flops);
}

TEST(SkalowanieLiczbyInstancji, Nczastek) {
    graf przestrzen = graf_lini(4, BEZ_NAZW);
    uint32_t liczba_czastek = 6;

    transformata_benchowana T = H;
    for (uint32_t j = 1; j < liczba_czastek; j++) {
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

    rezultat_benchu_2 rezultat("Linia N czastek");
    rezultat_benchu_2 min_pamiec("min-pamiec Linia N czastek");
    rezultat_benchu_2 min_flops("min-flops Linia N czastek");

    for (uint32_t i = 0; i < liczba_instancji; i++) {
        uint32_t liczba_iteracji = benchowane_liczby_iteracji[i];
        iteruj_spacer(spacer, rezultat, min_pamiec, min_flops, liczba_iteracji);
    }

    rezultaty_benchu.push_back(rezultat);
    rezultaty_benchu.push_back(min_pamiec);
    rezultaty_benchu.push_back(min_flops);
}

TEST(Koniec, PokazRezultat){
    okno_benchmarku okno("Liczba iteracji skalowanie");
    while (okno.tick_start()) {

        if (ImGui::Begin("Rezultaty")) {
            if (ImPlot::BeginPlot("##Rezultaty", ImVec2(800.0f, 800.0f))) {
                ImPlot::SetupAxes("liczba iteracji", "czas[ms]");

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
4 - typ spaceru, liczba czastek, liczba iteracji
5 - typ spaceru, spacer, liczby prac, liczby watkow na blok, liczba iteracji
*/



