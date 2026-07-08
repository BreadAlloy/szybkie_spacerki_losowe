#include <gtest/gtest.h>

#include "spacer_losowy.h"

#include "transformaty_wyspecializowane.h"

#include "definicje_typowych_macierzy.h"

#include "okno_benchmarku.h"
#include "grafika.h"

// ---=== CONFIG ===---

typedef zesp towar_benchowany;
typedef TMDQ transformata_benchowana;
#define TEMPLATY_BENCHOWANE <towar_benchowany, transformata_benchowana>
typedef spacer_losowy TEMPLATY_BENCHOWANE spacer_benchowany;

constexpr uint32_t liczba_iteracji = 50000;

constexpr uint64_t ile_prac_na_watek = 5; // ma³o aby by³o du¿o mo¿liwoœci na kolizje wielow¹tkow¹
constexpr uint32_t ile_watkow_na_blok_max = 256;

void iteruj_spacer_gpu(spacer_benchowany& spacer, uint32_t co_ile_zapisac = liczba_iteracji - 1) {
    CZAS_INIT
    CZAS_START
    printf("CUDA start\n");

    spacer.zbuduj_na_cuda();

    podzielone_iteracje_na_gpu(
        spacer, FP_JEDEN, liczba_iteracji,
        ile_prac_na_watek, ile_watkow_na_blok_max,
        co_ile_zapisac);

    spacer.zburz_na_cuda();

    CZAS_STOP
}

void iteruj_spacer_cpu(spacer_benchowany& spacer, uint32_t co_ile_zapisac = liczba_iteracji - 1) {
    CZAS_INIT
    CZAS_START
    printf("CPU start\n");

    for (uint32_t i = 0; i < liczba_iteracji; i++) {
        spacer.iteracja_na_cpu();
        if (i % co_ile_zapisac == 0) {
            spacer.zapisz_iteracje();
        }
        spacer.dokoncz_iteracje(dt);
    }
    
    CZAS_STOP
}

void sprawdz_identycznosc(
    spacer::dane_iteracji<towar_benchowany>* iteracja1, 
    spacer::dane_iteracji<towar_benchowany>* iteracja2 ){

    ASSERT_EQ(iteracja1->wartosci.rozmiar, iteracja2->wartosci.rozmiar);

    size_t rozmiar = iteracja1->wartosci.rozmiar;
    for(size_t i = 0; i < rozmiar; i++){
        towar_benchowany val1 = iteracja1->wartosci[i];
        towar_benchowany val2 = iteracja2->wartosci[i];

        ASSERT_NEAR(val1.Re, val2.Re, 1e-6);
        ASSERT_NEAR(val1.Im, val2.Im, 1e-6);
    }
}

void test_for_spacer(spacer_benchowany& spacer){
    spacer_benchowany spacer_cpu = spacer;
    spacer_benchowany spacer_gpu = spacer;

    iteruj_spacer_cpu(spacer_cpu);
    iteruj_spacer_gpu(spacer_gpu);

    ASSERT_EQ(spacer_cpu.iteracje_zapamietane.rozmiar, 2);
    ASSERT_EQ(spacer_gpu.iteracje_zapamietane.rozmiar, 2);

    sprawdz_identycznosc(
        spacer_cpu.iteracje_zapamietane[1],
        spacer_gpu.iteracje_zapamietane[1]);
}

TEST(CudaTakaJakCpu, Linia) {
    spacer_benchowany spacer = spacer_linia TEMPLATY_BENCHOWANE
    (2500, H, H);

    test_for_spacer(spacer);
}

TEST(CudaTakaJakCpu, Grid2d) {
    spacer_benchowany spacer = spacer_krata_2D TEMPLATY_BENCHOWANE
    (51, Fourier_4, Fourier_4);

    test_for_spacer(spacer);
}

TEST(CudaTakaJakCpu, Nczastek) {
    graf przestrzen = graf_lini(4, BEZ_NAZW);
    uint32_t liczba_czastek = 4;

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

    test_for_spacer(spacer);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    int ret =  RUN_ALL_TESTS();
    if(ret != 0){
        okno_benchmarku okno("Spojrzenie reczne");  
 
        graf przestrzen(graf_krata_2D(51));
        spacer_benchowany spacer = spacer_krata_2D TEMPLATY_BENCHOWANE
        (51, Fourier_4, Fourier_4, &przestrzen);

        spacer_benchowany spacer_cpu = spacer;
        spacer_benchowany spacer_gpu = spacer;

        constexpr uint32_t ile_zapisac = 500;

        iteruj_spacer_cpu(spacer_cpu, liczba_iteracji / ile_zapisac);
        iteruj_spacer_gpu(spacer_gpu, liczba_iteracji / ile_zapisac);

        std::vector<grafika*> grafiki_cpu;
        std::vector<grafika*> grafiki_gpu;
        std::vector<grafika*> grafiki_diff;

        grafiki_cpu.resize(ile_zapisac);
        grafiki_gpu.resize(ile_zapisac);
        grafiki_diff.resize(ile_zapisac);

        for (uint32_t i = 0; i < ile_zapisac; i++) {
            spacer::dane_iteracji<towar_benchowany>* iteracja_cpu = spacer_cpu.iteracje_zapamietane[i];
            grafiki_cpu[i] = grafika_P_kierunkow_dla_kraty_2D(spacer_cpu, *iteracja_cpu,
                51, 51);

            spacer::dane_iteracji<towar_benchowany>* iteracja_gpu = spacer_gpu.iteracje_zapamietane[i];
            grafiki_gpu[i] = grafika_P_kierunkow_dla_kraty_2D(spacer_gpu, *iteracja_gpu,
                51, 51);

            grafika* grafika_diff = new grafika(51, 51);
            for(ID_W i = 0; i < spacer.trwale.liczba_wierzcholkow(); i++){
                spacer::wierzcholek& W = spacer.trwale.wierzcholki[i];
                bool error = false;
                for(ID_K k = 0; k < W.liczba_kierunkow; k++){
                    towar_benchowany gpu_war = iteracja_gpu->wartosci[W.start_wartosci + k];
                    towar_benchowany cpu_war = iteracja_cpu->wartosci[W.start_wartosci + k];

                    prob_t diff = abs(P(gpu_war) - P(cpu_war));
                    if(diff > 1e-6f){
                        error |= true;
                    }
                }

                if(error){
                    grafika_diff->data[4 * i + 0] = (uint8_t)0xFF;

                } else {
                    grafika_diff->data[4 * i + 0] = (uint8_t)0;
                }
                grafika_diff->data[4 * i + 1] = (uint8_t)0;
                grafika_diff->data[4 * i + 2] = (uint8_t)0;
                grafika_diff->data[4 * i + 3] = (uint8_t)0xFF;
            }
            grafika_diff->LoadTextureFromMemory();

            grafiki_diff[i] = grafika_diff;
        }
        int ogladany_czas = 0;
        while (okno.tick_start()) {

            if (ImGui::Begin("CudaTakaJakCpu")) {

                ImGui::SliderInt("Ktora grafika jest pokazywana", &ogladany_czas, 0, ile_zapisac-1);

                plot_spacer_dla_kraty_2D(spacer_cpu, ogladany_czas, przestrzen,
                    grafiki_cpu[ogladany_czas], 51, 51,
                    2.0f, "Spacer CPU");
                
                ImGui::SameLine();

                {
                    ImVec2 bmin(0.0, 0.0);
                    ImVec2 bmax((float)51, (float)51);
                    ImVec2 uv0(0.0, 0.0);
                    ImVec2 uv1(1.0, -1.0); // bo tak tworze osie przy tworzeniu grafu

                    if (ImPlot::BeginPlot("diff", ImVec2(2.0f * 200.0f, 2.0f * 200.0f))) {
                        ImPlot::SetupAxes("pozycja X", "pozycja Y");
                        ImPlot::PlotImage("##diff",
                            (ImTextureID)(intptr_t)(grafiki_diff[ogladany_czas]->texture),
                            bmin, bmax, uv0, uv1);
                        ImPlot::EndPlot();
                    }
                }

                ImGui::SameLine();

                plot_spacer_dla_kraty_2D(spacer_gpu, ogladany_czas, przestrzen,
                    grafiki_gpu[ogladany_czas], 51, 51,
                    2.0f, "Spacer GPU");

                ImGui::End();
            }

            okno.tick_finish();
        }

        for (auto g : grafiki_cpu) delete g;
        for (auto g : grafiki_gpu) delete g;
        for (auto g : grafiki_diff) delete g;

    }
    return ret;
}




