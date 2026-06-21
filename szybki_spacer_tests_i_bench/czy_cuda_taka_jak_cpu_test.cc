#include <gtest/gtest.h>

#include "spacer_losowy.h"

#include "transformaty_wyspecializowane.h"

#include "definicje_typowych_macierzy.h"

// ---=== CONFIG ===---

typedef zesp towar_benchowany;
typedef TMDQ transformata_benchowana;
#define TEMPLATY_BENCHOWANE <towar_benchowany, transformata_benchowana>
typedef spacer_losowy TEMPLATY_BENCHOWANE spacer_benchowany;

constexpr uint32_t liczba_iteracji = 50000;

constexpr uint64_t ile_prac_na_watek = 5; // ma³o aby by³o du¿o mo¿liwoœci na kolizje wielow¹tkow¹
constexpr uint32_t ile_watkow_na_blok_max = 256;

void iteruj_spacer_gpu(spacer_benchowany& spacer) {
    CZAS_INIT
    CZAS_START
    printf("CUDA start\n");

    spacer.zbuduj_na_cuda();

    proste_iteracje_na_gpu(
        spacer, FP_JEDEN, liczba_iteracji,
        ile_prac_na_watek, ile_watkow_na_blok_max,
        liczba_iteracji - 1);

    spacer.zburz_na_cuda();

    CZAS_STOP
}

void iteruj_spacer_cpu(spacer_benchowany& spacer) {
    CZAS_INIT
    CZAS_START
    printf("CPU start\n");

    for (uint32_t i = 0; i < liczba_iteracji; i++) {
        spacer.iteracja_na_cpu();
        if (i % (liczba_iteracji - 1) == 0) {
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

        ASSERT_NEAR(val1.Re, val2.Re, 10e-4);
        ASSERT_NEAR(val1.Im, val2.Im, 10e-4);
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
    (50, Fourier_4, Fourier_4);

    test_for_spacer(spacer);
}

TEST(SkalowanieLiczbyInstancji, Nczastek) {
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
    return RUN_ALL_TESTS();
}




