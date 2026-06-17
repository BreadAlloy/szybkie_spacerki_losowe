#pragma once
#include "spacer_losowy.h"
#include "alg_liniowa.h"
#include "transformaty_wyspecializowane.h"
#include "definicje_typowych_macierzy.h"
#include "grafika.h"
#include "zapamietywacz.h"

constexpr double pi = 3.1415926535897932;

#define lengthof(cos) (sizeof(cos)/sizeof(cos[0]))

static __host__  void przejrzenie_stanow_poczatkowych(){
	uint64_t ile_przejrzec = 8;
	uint32_t liczba_wierzcholkow_boku = 101;
	uint64_t ile_razy_zapisz = 4; // tak naprawde 5
	uint64_t liczba_iteracji = 200;
	spacer_losowy<zesp, TMCQ> spacer(spacer_krata_2D_cykl<zesp, TMCQ>(liczba_wierzcholkow_boku, Fourier_4, nullptr));

	spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = zero(zesp());

	spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(zesp()) / sqrt(2.0);
	spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci + 2] = jeden(zesp()) / sqrt(2.0);

	printf("Jest %llu do przejrzenia\n", ile_przejrzec);

	std::string folder = "stan_poczatkowy";
	{
		std::string komentarz = "KOMANTARZ:\n\
Przejrzenie mo¿liwych operatorów powsta³ych z tenosra dwóch operatorów 2 na 2. Spacer na grafie kraty cyklicznej.\n\
Nie ma sprawdzania podobienstwa opertatora do poprzednich. dt = 0.1.\n\
Te wczesniejsz wszystkie tak samo wygl¹daj¹.\n\
Plik transformaty.csv zawiera przejrzane transforamaty, indeks w pliku odpowiada temu w nazwie grafiki.\n\
Kod u¿yty do wygenerowania(wywo³ywane funkcje mog¹ siê zmieniæ):\n\n\n";

		FILE* fd = fopen((folder + "\\komentarz.txt").c_str(), "wbx"); // to powninno failowaæ jeœli ju¿ istnieje
		ASSERT_Z_ERROR_MSG(fd != nullptr, "Cos nie tak z plikiem\n");

		fwrite(komentarz.c_str(), sizeof(char), komentarz.length(), fd);

		FILE* kod_generujacy = fopen("zadanie.h", "rb");
		ASSERT_Z_ERROR_MSG(kod_generujacy != nullptr, "Cos nie tak z plikiem\n");

		int numread = 1;
		char* buffer[1024];
		while(numread != 0){
			numread = fread(buffer, sizeof(char), 1024, kod_generujacy);
			fwrite(buffer, sizeof(char), numread, fd);
		}

		fclose(kod_generujacy);
		fclose(fd);
	}

	spacer_losowy<zesp, TMCQ> przegladany(spacer);

	for(uint64_t i = 0; i < ile_przejrzec; i++){			
		przegladany = spacer;
		double kat = ((double)i / (double)ile_przejrzec) * 2.0 * pi;

		double x = cos(kat);
		double y = sin(kat);

		przegladany.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = MAX(x, 0.0);
		przegladany.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci + 1] = MIN(x, 0.0);
		przegladany.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci + 2] = MAX(y, 0.0);
		przegladany.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci + 3] = MIN(y, 0.0);

		printf("Iteracja %lld\n", i);
		CZAS_INIT
		przegladany.zbuduj_na_cuda();

		printf("GPU start\n");
		CZAS_START
		iteracje_na_gpu<zesp, TMCQ>(przegladany, dt, liczba_iteracji, 30, 500, (liczba_iteracji - 1UL) / ile_razy_zapisz, 1, liczba_iteracji + 1);
		CZAS_STOP
		printf("GPU koniec\n");

		przegladany.zburz_na_cuda();

		for(uint64_t j = 0; j <= ile_razy_zapisz; j++){
			grafika* G = grafika_P_kierunkow_dla_kraty_2D<zesp, TMCQ>(przegladany, *(przegladany.iteracje_zapamietane[j]),
			liczba_wierzcholkow_boku, liczba_wierzcholkow_boku, nullptr, 1.0f, false);

			G->SaveToFile(folder + "//transformata_" + std::to_string(i) + "-grafika_" + std::to_string(j) + ".bmp");
			delete G;
		}
	}
}


