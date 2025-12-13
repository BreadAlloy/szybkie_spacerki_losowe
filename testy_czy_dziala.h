#pragma once

#undef IM_ASSERT
#define IM_ASSERT(_EXPR)  lepszy_assert(_EXPR)
#include "imgui_internal.h"

#include "spacer_losowy.h"

#include "grafika.h"

#include "transformaty_wyspecializowane.h"

#include "definicje_typowych_macierzy.h"

namespace ImPlot{
template <typename T>
void PlotLineLepsze(const char* label_id, const double* xs, const T* ys, int count, ImPlotLineFlags flags, int offset, int stride);
};

struct test_spaceru_klasyczny_dyskretny{
	std::vector<grafika*> grafiki_iteracji;
	grafika* celownik = nullptr;
	std::string nazwa_okna;

	uint32_t height = 0;
	uint32_t width = 0;

	double ostatni_czas_odswiezenia = glfwGetTime();
	float okres_pokazu_slajdow = 1.0f;

	int pokazywana_grafika = 0;
	float skala_obrazu = 1.0f;

	double macierz3x3[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
	double macierz4x4[16] = { 0.01, 0.97, 0.01, 0.01,
							 0.97, 0.01, 0.01, 0.01,
							 0.01, 0.01, 0.01, 0.97,
							 0.01, 0.01, 0.97, 0.01 };

	const uint32_t liczba_wierzcholkow_boku = 21;
	const uint32_t liczba_iteracji = 100;
	const uint32_t jak_czesto_zapisac = 1;

	graf przestrzen;
	spacer_losowy<double, TMDK> spacer;

	__host__ test_spaceru_klasyczny_dyskretny()
		: nazwa_okna("Test spaceru klasyczny dyskretny")
		, przestrzen(graf_krata_2D(liczba_wierzcholkow_boku))
		, spacer(spacer_krata_2D<double, TMDK>(liczba_wierzcholkow_boku, transformata_macierz<double>(4, macierz4x4), transformata_macierz<double>(3, macierz3x3), transformata_macierz<double>(1.0, 0.0, 0.0, 1.0), &przestrzen)){
		
		spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = 0.5;
		spacer.iteracjaA[spacer.trwale.wierzcholki[((liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2) + 1].start_wartosci + 2] = 0.5;

		spacer.zapisz_iteracje();
		for (uint64_t i = 0; i < liczba_iteracji; i++) {
			spacer.iteracja_na_cpu();
			spacer.dokoncz_iteracje(1.0);
			if((i % jak_czesto_zapisac) == 0){
				spacer.zapisz_iteracje();
			}
		}

		celownik = new grafika("textures/crosshair.png");
		ASSERT_Z_ERROR_MSG(celownik->texture != 0, "Cos nie tak z textura celownika\n");

		przygotuj_grafiki();
	}

	__host__ void przygotuj_grafiki() {
		height = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		width = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		ASSERT_Z_ERROR_MSG(height * width == spacer.trwale.liczba_wierzcholkow(), "Tego spaceru nie da sie przedstawic jako kwadrat\n");

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<double>& iteracja = *(spacer.iteracje_zapamietane[i]);
			grafika* G = new grafika(width, height);
			double max = 0.0;

			for (uint64_t j = 0; j < spacer.trwale.liczba_wierzcholkow(); j++) {
				double prawdopodobienstwo = 0.0;
				spacer::wierzcholek& wierzcholek = spacer.trwale.wierzcholki[j];

				for (uint8_t k = 0; k < wierzcholek.liczba_kierunkow; k++) {
					prawdopodobienstwo += P(iteracja.wartosci[wierzcholek.start_wartosci + k]);
				}

				if (prawdopodobienstwo > max) {
					max = prawdopodobienstwo;
				}

				float* ptr = (float*)(G->data + 4 * j);
				*ptr = (float)(prawdopodobienstwo * 255.0);
				//(G->data)[4 * j + 0] = jasnosc;  // R
				//(G->data)[4 * j + 1] = jasnosc;  // G
				//(G->data)[4 * j + 2] = jasnosc;  // B
				//(G->data)[4 * j + 3] = 0xFF;
			}

			for (uint64_t j = 0; j < spacer.trwale.liczba_wierzcholkow(); j++) {
				uint8_t jasnosc_znormalizowana = (uint8_t)((*(float*)((G->data) + 4 * j)) / max);

				(G->data)[4 * j + 0] = jasnosc_znormalizowana;  // R
				(G->data)[4 * j + 1] = jasnosc_znormalizowana;  // G
				(G->data)[4 * j + 2] = jasnosc_znormalizowana;	// B
				(G->data)[4 * j + 3] = 0xFF;
			}

			G->LoadTextureFromMemory();
			grafiki_iteracji.push_back(G);

		}
	}

	__host__ void display_image(ImGuiIO&) {
		grafika* G = grafiki_iteracji[pokazywana_grafika];
		ImGui::Text("t = %lf", spacer.iteracje_zapamietane[pokazywana_grafika]->czas);
		ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, 0.01f, 1.0f);

		plot_grafike_dla_kraty_2D(spacer, pokazywana_grafika, przestrzen, G, width, height, skala_obrazu);

		if (okres_pokazu_slajdow < 0.95f) {
			double czas = glfwGetTime();
			if (czas > (ostatni_czas_odswiezenia + (double)okres_pokazu_slajdow)) {
				ostatni_czas_odswiezenia = czas;
				pokazywana_grafika = (pokazywana_grafika + 1) % grafiki_iteracji.size();
			}
		}
		else {
			ostatni_czas_odswiezenia = glfwGetTime();
		}
	}

	__host__ void pokaz_okno(ImGuiIO& io) {
		ImGui::Begin(nazwa_okna.c_str());
		ImGui::SliderFloat("Rozmiar grafiki", &skala_obrazu, 0.0f, 10.0f);
		ImGui::SliderInt("Ktora grafika jest pokazywana", &pokazywana_grafika, 0, (int)grafiki_iteracji.size() - 1);
		display_image(io);
		ImGui::End();
	}

	__host__ ~test_spaceru_klasyczny_dyskretny() {
		delete celownik;
		for (auto g : grafiki_iteracji) {
			delete g;
		}
	}
};


struct test_spaceru_kwantowy_dyskretny {
	std::vector<grafika*> grafiki_iteracji;
	grafika* celownik = nullptr;
	std::string nazwa_okna;

	uint32_t height = 0;
	uint32_t width = 0;

	double ostatni_czas_odswiezenia = glfwGetTime();
	float okres_pokazu_slajdow = 1.0f;

	int pokazywana_grafika = 0;
	float skala_obrazu = 1.0f;

	std::vector<zesp> katy;
	std::vector<double> katy_norm;
	std::vector<double> prawdopodop;
	std::vector<double> czasy;

	const uint32_t liczba_wierzcholkow_boku = 201;
	const uint32_t liczba_iteracji = 5000;
	const uint32_t jak_czesto_zapisac = 1;

	graf przestrzen;
	spacer_losowy<zesp, TMDQ> spacer;

	__host__ test_spaceru_kwantowy_dyskretny()
		: nazwa_okna("Test spaceru kwantowego dyskretny")
		, przestrzen(graf_krata_2D(liczba_wierzcholkow_boku))
		, spacer(spacer_krata_2D<zesp, TMDQ>(liczba_wierzcholkow_boku,
		mnoz(HxH, std_kierunki_krata), I_3, I_2, &przestrzen)) {

		spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(zesp()) / std::sqrt(2.0);
		spacer.iteracjaA[spacer.trwale.wierzcholki[((liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2) + 1].start_wartosci + 2] = jeden(zesp()) / std::sqrt(2.0);

		spacer.zapisz_iteracje();
		for (uint64_t i = 1; i < liczba_iteracji; i++) {
			spacer.iteracja_na_cpu();
			spacer.dokoncz_iteracje(1.0);
			if ((i % jak_czesto_zapisac) == 0) {
				spacer.zapisz_iteracje();
			}
		}

		celownik = new grafika("textures/crosshair.png");
		ASSERT_Z_ERROR_MSG(celownik->texture != 0, "Cos nie tak z textura celownika\n");

		przygotuj_grafiki();
	}

	__host__ uint64_t liczba_zapamietanych_iteracji(){
		return spacer.iteracje_zapamietane.rozmiar;
	}

	__host__ void przygotuj_grafiki() {
		height = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		width = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		ASSERT_Z_ERROR_MSG(height * width == spacer.trwale.liczba_wierzcholkow(), "Tego spaceru nie da sie przedstawic jako kwadrat\n");

		grafiki_iteracji.resize(liczba_zapamietanych_iteracji());
		katy.resize(liczba_zapamietanych_iteracji());
		katy_norm.resize(liczba_zapamietanych_iteracji());
		prawdopodop.resize(liczba_zapamietanych_iteracji());
		czasy.resize(liczba_zapamietanych_iteracji());

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[i]);
			grafiki_iteracji[i] = grafika_P_dla_kraty_2D(spacer, iteracja,
				width, height, &(prawdopodop[i]));
			czasy[i] = iteracja.czas;
		}

		spacer::dane_iteracji<zesp>& iteracja_pierwsza = *(spacer.iteracje_zapamietane[0]);

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[i]);
			katy[i] = dot(iteracja_pierwsza.wartosci, iteracja.wartosci);
			katy_norm[i] = std::sqrt(katy[i].norm());
		}
	}

	__host__ void display_image(ImGuiIO&) {
		grafika* G = grafiki_iteracji[pokazywana_grafika];
		ImGui::Text("t = %lf", spacer.iteracje_zapamietane[pokazywana_grafika]->czas);
		ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, 0.01f, 1.0f);

		plot_grafike_dla_kraty_2D(spacer, pokazywana_grafika, przestrzen, G, width, height, skala_obrazu);

		if (okres_pokazu_slajdow < 0.95f) {
			double czas = glfwGetTime();
			if (czas > (ostatni_czas_odswiezenia + (double)okres_pokazu_slajdow)) {
				ostatni_czas_odswiezenia = czas;
				pokazywana_grafika = (pokazywana_grafika + 1) % grafiki_iteracji.size();
			}
		}
		else {
			ostatni_czas_odswiezenia = glfwGetTime();
		}
	}

	__host__ void pokaz_wykresy(ImGuiIO&){
		if (ImPlot::BeginPlot("##Dane w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
			ImPlot::PlotInfLines("Vertical pomocnik", &czasy[pokazywana_grafika], 1);
			ImPlot::PlotLine("Prawdopodobienstwa suma", czasy.data(), prawdopodop.data(), (int)liczba_zapamietanych_iteracji());
			ImPlot::PlotLineLepsze("Kat Re", czasy.data(), (double*)katy.data(), (int)liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLineLepsze("Kat Im", czasy.data(), (double*)katy.data()+1, (int)liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLine("Kat norma", czasy.data(), katy_norm.data(), (int)liczba_zapamietanych_iteracji());
			ImPlot::EndPlot();
		}
	}

	__host__ void pokaz_okno(ImGuiIO& io) {
		ImGui::Begin(nazwa_okna.c_str());
		ImGui::SliderFloat("Rozmiar grafiki", &skala_obrazu, 0.0f, 10.0f);
		ImGui::SliderInt("Ktora grafika jest pokazywana", &pokazywana_grafika, 0, (int)liczba_zapamietanych_iteracji() - 1);
		display_image(io);
		ImGui::SameLine();
		pokaz_wykresy(io);
		ImGui::End();
	}

	__host__ ~test_spaceru_kwantowy_dyskretny() {
		delete celownik;
		for (auto g : grafiki_iteracji) {
			delete g;
		}
	}
};

struct test_spaceru_kwantowy_dyskretny_gpu {
	std::vector<grafika*> grafiki_iteracji;
	grafika* celownik = nullptr;
	std::string nazwa_okna;

	uint32_t height = 0;
	uint32_t width = 0;

	double ostatni_czas_odswiezenia = glfwGetTime();
	float okres_pokazu_slajdow = 1.0f;

	int pokazywana_grafika = 0;
	float skala_obrazu = 1.0f;

	std::vector<zesp> katy;
	std::vector<double> katy_norm;
	std::vector<double> prawdopodop;
	std::vector<double> czasy;

	const uint32_t liczba_wierzcholkow_boku = 101;
	const uint32_t liczba_iteracji = 200;
	const uint32_t jak_czesto_zapisac = 5;

	graf przestrzen;
	spacer_losowy<zesp, TMDQ> spacer;

	__host__ test_spaceru_kwantowy_dyskretny_gpu()
		: nazwa_okna("Test spaceru kwantowego dyskretny GPU")
		, przestrzen(graf_krata_2D(liczba_wierzcholkow_boku))
		, spacer(spacer_krata_2D<zesp, TMDQ>(liczba_wierzcholkow_boku, HxH,
		 I_3, I_2, &przestrzen)) {

		spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(zesp()) / std::sqrt(2.0);
		spacer.iteracjaA[spacer.trwale.wierzcholki[((liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2) + 1].start_wartosci + 2] = jeden(zesp()) / std::sqrt(2.0);

		printf("CUDA start\n");
		spacer.zbuduj_na_cuda();

		spacer.zapisz_iteracje();
		for (uint64_t i = 1; i < liczba_iteracji; i += jak_czesto_zapisac) {
			//for(uint64_t j = 0; j < spacer.trwale.ile_watkow(10); j++){
			//	symulowana_iteracja_na_gpu<zesp, TMDQ>(&spacer, j);
			//}
			//spacer.dokoncz_iteracje(1.0);
			iteruj_na_gpu<zesp, TMDQ>(spacer, jak_czesto_zapisac);
			spacer.cuda_przynies();
			spacer.zapisz_iteracje();
		}

		spacer.zburz_na_cuda();
		printf("CUDA koniec\n");

		celownik = new grafika("textures/crosshair.png");
		ASSERT_Z_ERROR_MSG(celownik->texture != 0, "Cos nie tak z textura celownika\n");

		przygotuj_grafiki();
	}

	__host__ uint64_t liczba_zapamietanych_iteracji() {
		return spacer.iteracje_zapamietane.rozmiar;
	}

	__host__ void przygotuj_grafiki() {
		height = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		width = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		ASSERT_Z_ERROR_MSG(height * width == spacer.trwale.liczba_wierzcholkow(), "Tego spaceru nie da sie przedstawic jako kwadrat\n");

		grafiki_iteracji.resize(liczba_zapamietanych_iteracji());
		katy.resize(liczba_zapamietanych_iteracji());
		katy_norm.resize(liczba_zapamietanych_iteracji());
		prawdopodop.resize(liczba_zapamietanych_iteracji());
		czasy.resize(liczba_zapamietanych_iteracji());

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[i]);
			grafiki_iteracji[i] = grafika_P_dla_kraty_2D(spacer, iteracja,
				width, height, &(prawdopodop[i]));
			czasy[i] = iteracja.czas;
		}

		spacer::dane_iteracji<zesp>& iteracja_pierwsza = *(spacer.iteracje_zapamietane[0]);

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[i]);
			katy[i] = dot(iteracja_pierwsza.wartosci, iteracja.wartosci);
			katy_norm[i] = std::sqrt(katy[i].norm());
		}
	}

	__host__ void display_image(ImGuiIO&) {
		grafika* G = grafiki_iteracji[pokazywana_grafika];
		ImGui::Text("t = %lf", spacer.iteracje_zapamietane[pokazywana_grafika]->czas);
		ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, 0.01f, 1.0f);

		plot_grafike_dla_kraty_2D(spacer, pokazywana_grafika, przestrzen, G, width, height, skala_obrazu);

		if (okres_pokazu_slajdow < 0.95f) {
			double czas = glfwGetTime();
			if (czas > (ostatni_czas_odswiezenia + (double)okres_pokazu_slajdow)) {
				ostatni_czas_odswiezenia = czas;
				pokazywana_grafika = (pokazywana_grafika + 1) % grafiki_iteracji.size();
			}
		}
		else {
			ostatni_czas_odswiezenia = glfwGetTime();
		}
	}

	__host__ void pokaz_wykresy(ImGuiIO&) {
		if (ImPlot::BeginPlot("##Dane w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
			ImPlot::PlotInfLines("Vertical pomocnik", &czasy[pokazywana_grafika], 1);
			ImPlot::PlotLine("Prawdopodobienstwa suma", czasy.data(), prawdopodop.data(), (int)liczba_zapamietanych_iteracji());
			ImPlot::PlotLineLepsze("Kat Re", czasy.data(), (double*)katy.data(), (int)liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLineLepsze("Kat Im", czasy.data(), (double*)katy.data() + 1, (int)liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLine("Kat norma", czasy.data(), katy_norm.data(), (int)liczba_zapamietanych_iteracji());
			ImPlot::EndPlot();
		}
	}

	__host__ void pokaz_okno(ImGuiIO& io) {
		ImGui::Begin(nazwa_okna.c_str());
		ImGui::SliderFloat("Rozmiar grafiki", &skala_obrazu, 0.0f, 10.0f);
		ImGui::SliderInt("Ktora grafika jest pokazywana", &pokazywana_grafika, 0, (int)liczba_zapamietanych_iteracji() - 1);
		display_image(io);
		ImGui::SameLine();
		pokaz_wykresy(io);
		ImGui::End();
	}

	__host__ ~test_spaceru_kwantowy_dyskretny_gpu() {
		delete celownik;
		for (auto g : grafiki_iteracji) {
			delete g;
		}
	}
};

struct test_spaceru_kwantowy_ciagly {
	std::vector<grafika*> grafiki_iteracji;
	grafika* celownik = nullptr;
	std::string nazwa_okna;

	uint32_t height = 0;
	uint32_t width = 0;

	double ostatni_czas_odswiezenia = glfwGetTime();
	float okres_pokazu_slajdow = 1.0f;

	int pokazywana_grafika = 0;
	float skala_obrazu = 1.0f;

	std::vector<zesp> katy;
	std::vector<double> katy_norm;
	std::vector<double> prawdopodop;
	std::vector<double> czasy;

	zesp macierz3x3[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
	zesp macierz4x4[16] = { 0.5,  0.5,  0.5,  0.5,
							0.5, -0.5,  0.5, -0.5,
							0.5,  0.5, -0.5, -0.5,
							0.5, -0.5, -0.5,  0.5 };

	const uint32_t liczba_wierzcholkow_boku = 21;
	const uint32_t liczba_iteracji = 100000;
	const uint32_t jak_czesto_zapisac = 25;

	graf przestrzen;
	spacer_losowy<zesp, TMCQ> spacer;

	__host__ test_spaceru_kwantowy_ciagly()
		: nazwa_okna("Test spaceru kwantowego ciagly")
		, przestrzen(graf_krata_2D(liczba_wierzcholkow_boku))
		, spacer(spacer_krata_2D<zesp, TMCQ>(liczba_wierzcholkow_boku, transformata_macierz<zesp>(4, macierz4x4), transformata_macierz<zesp>(3, macierz3x3), transformata_macierz<zesp>(1.0, 0.0, 0.0, 1.0), &przestrzen)) {

		spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(zesp()) / std::sqrt(2.0);
		spacer.iteracjaA[spacer.trwale.wierzcholki[((liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2) + 1].start_wartosci + 2] = jeden(zesp()) / std::sqrt(2.0);

		spacer.zapisz_iteracje();
		for (uint64_t i = 1; i < liczba_iteracji; i++) {
			spacer.iteracja_na_cpu();
			spacer.dokoncz_iteracje(dt);
			if ((i % jak_czesto_zapisac) == 0) {
				spacer.zapisz_iteracje();
			}
		}

		celownik = new grafika("textures/crosshair.png");
		ASSERT_Z_ERROR_MSG(celownik->texture != 0, "Cos nie tak z textura celownika\n");

		przygotuj_grafiki();
	}

	__host__ uint64_t liczba_zapamietanych_iteracji() {
		return spacer.iteracje_zapamietane.rozmiar;
	}

	__host__ void przygotuj_grafiki() {
		height = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		width = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		ASSERT_Z_ERROR_MSG(height * width == spacer.trwale.liczba_wierzcholkow(), "Tego spaceru nie da sie przedstawic jako kwadrat\n");

		grafiki_iteracji.resize(liczba_zapamietanych_iteracji());
		katy.resize(liczba_zapamietanych_iteracji());
		katy_norm.resize(liczba_zapamietanych_iteracji());
		prawdopodop.resize(liczba_zapamietanych_iteracji());
		czasy.resize(liczba_zapamietanych_iteracji());

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[i]);
			grafiki_iteracji[i] = grafika_P_dla_kraty_2D(spacer, iteracja,
										 width, height, &(prawdopodop[i]));
			czasy[i] = iteracja.czas;
		}

		spacer::dane_iteracji<zesp>& iteracja_pierwsza = *(spacer.iteracje_zapamietane[0]);

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[i]);
			katy[i] = dot(iteracja_pierwsza.wartosci, iteracja.wartosci);
			katy_norm[i] = std::sqrt(katy[i].norm());
		}
	}

	__host__ void display_image(ImGuiIO&) {
		grafika* G = grafiki_iteracji[pokazywana_grafika];
		ImGui::Text("t = %lf", spacer.iteracje_zapamietane[pokazywana_grafika]->czas);
		ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, 0.01f, 1.0f);

		plot_grafike_dla_kraty_2D(spacer, pokazywana_grafika, przestrzen, G, width, height, skala_obrazu);

		if (okres_pokazu_slajdow < 0.95f) {
			double czas = glfwGetTime();
			if (czas > (ostatni_czas_odswiezenia + (double)okres_pokazu_slajdow)) {
				ostatni_czas_odswiezenia = czas;
				pokazywana_grafika = (pokazywana_grafika + 1) % grafiki_iteracji.size();
			}
		}
		else {
			ostatni_czas_odswiezenia = glfwGetTime();
		}
	}

	__host__ void pokaz_wykresy(ImGuiIO&) {
		if (ImPlot::BeginPlot("##Dane w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
			ImPlot::PlotInfLines("Vertical pomocnik", &czasy[pokazywana_grafika], 1);
			ImPlot::PlotLine("Prawdopodobienstwa suma", czasy.data(), prawdopodop.data(), (int)liczba_zapamietanych_iteracji());
			ImPlot::PlotLineLepsze("Kat Re", czasy.data(), (double*)katy.data(), (int)liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLineLepsze("Kat Im", czasy.data(), (double*)katy.data() + 1, (int)liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLine("Kat norma", czasy.data(), katy_norm.data(), (int)liczba_zapamietanych_iteracji());
			ImPlot::EndPlot();
		}
	}

	__host__ void pokaz_okno(ImGuiIO& io) {
		ImGui::Begin(nazwa_okna.c_str());
		ImGui::SliderFloat("Rozmiar grafiki", &skala_obrazu, 0.0f, 10.0f);
		ImGui::SliderInt("Ktora grafika jest pokazywana", &pokazywana_grafika, 0, (int)liczba_zapamietanych_iteracji() - 1);
		display_image(io);
		ImGui::SameLine();
		pokaz_wykresy(io);
		ImGui::End();
	}

	__host__ ~test_spaceru_kwantowy_ciagly() {
		delete celownik;
		for (auto g : grafiki_iteracji) {
			delete g;
		}
	}
};


struct test_czasow_wykonania_kwantowy {
	std::string nazwa_okna;

	uint32_t height = 0;
	uint32_t width = 0;

	double ostatni_czas_odswiezenia = glfwGetTime();
	float okres_pokazu_slajdow = 1.0f;

	int pokazywana_grafika = 0;
	float skala_obrazu = 1.0f;

	const uint32_t liczba_wierzcholkow_boku = 3001;
	uint32_t liczba_iteracji = 10;
	uint32_t ile_pracy_na_jeden_watek = 100;
	uint32_t max_liczba_watkow_w_bloku = 100;

	graf przestrzen;
	spacer_losowy<zesp, TMDQ> spacer_benchowany;

	spacer_losowy<zesp, TMDQ> spacer_cpu;
	uint64_t czas_cpu_ys = 0;
	std::vector<grafika*> grafiki_iteracji_cpu;

	spacer_losowy<zesp, TMDQ> spacer_gpu;
	uint64_t czas_gpu_ys = 0;
	std::vector<grafika*> grafiki_iteracji_gpu;

	__host__ test_czasow_wykonania_kwantowy()
		: nazwa_okna("Test czasow wykonania kwantowy")
		, przestrzen(graf_krata_2D(liczba_wierzcholkow_boku))
		, spacer_benchowany(spacer_krata_2D<zesp, TMDQ>(liczba_wierzcholkow_boku,
		 tensor(X, H), I_3, I_2, &przestrzen))
		, spacer_cpu(spacer_benchowany)
		, spacer_gpu(spacer_benchowany){

		spacer_benchowany.iteracjaA[spacer_benchowany.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(zesp()) / std::sqrt(2.0);
		spacer_benchowany.iteracjaA[spacer_benchowany.trwale.wierzcholki[((liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2) + 1].start_wartosci + 2] = jeden(zesp()) / std::sqrt(2.0);

		spacer_benchowany.zapisz_iteracje();

		grafiki_iteracji_cpu.resize(2);
		grafiki_iteracji_gpu.resize(2);

		height = (uint32_t)std::sqrt(spacer_benchowany.trwale.liczba_wierzcholkow());
		width = (uint32_t)std::sqrt(spacer_benchowany.trwale.liczba_wierzcholkow());
		ASSERT_Z_ERROR_MSG(height * width == spacer_benchowany.trwale.liczba_wierzcholkow(), "Tego spaceru nie da sie przedstawic jako kwadrat\n");

		przelicz_cpu();
		przelicz_gpu();
	}

	__host__ uint64_t liczba_zapamietanych_iteracji() {
		return spacer_cpu.iteracje_zapamietane.rozmiar;
	}

	__host__ void przelicz_cpu(){
		printf("\nLiczba iteracji:%d\n", liczba_iteracji);
		spacer_cpu = spacer_benchowany;

		CZAS_INIT
		printf("CPU start\n");
		CZAS_START
		for (uint64_t i = 0; i < liczba_iteracji; i++) {
			spacer_cpu.iteracja_na_cpu();
			spacer_cpu.dokoncz_iteracje(1.0);
		}
		CZAS_STOP
		printf("CPU koniec\n");
		czas_cpu_ys = diff;
		spacer_cpu.zapisz_iteracje();

		przygotuj_grafiki_cpu();
	}

	__host__ void przelicz_gpu() {
		printf("\nLiczba iteracji:%d, Max liczba watkow w bloku:%d, Ile pracy na watek:%d\n", liczba_iteracji, max_liczba_watkow_w_bloku, ile_pracy_na_jeden_watek);
		
		CZAS_INIT
		spacer_gpu = spacer_benchowany;
		spacer_gpu.zbuduj_na_cuda();
		printf("GPU start\n");
		CZAS_START
		iteracje_na_gpu<zesp, TMDQ>(spacer_gpu, liczba_iteracji, ile_pracy_na_jeden_watek, max_liczba_watkow_w_bloku);
		CZAS_STOP
		printf("GPU koniec\n");
		czas_gpu_ys = diff;
		spacer_gpu.cuda_przynies();
		spacer_gpu.zapisz_iteracje();
		spacer_gpu.zburz_na_cuda();
	
		przygotuj_grafiki_gpu();
	}

	__host__ void przygotuj_grafiki_cpu() {
		for(auto& g : grafiki_iteracji_cpu){
			delete g;
		}

		for (uint64_t i = 0; i < spacer_cpu.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer_cpu.iteracje_zapamietane[i]);
			grafiki_iteracji_cpu[i] = grafika_P_dla_kraty_2D(spacer_cpu, 
											iteracja, width, height);
		}
	}

	__host__ void przygotuj_grafiki_gpu() {
		for (auto& g : grafiki_iteracji_gpu) {
			delete g;
		}

		for (uint64_t i = 0; i < spacer_gpu.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer_gpu.iteracje_zapamietane[i]);
			grafiki_iteracji_gpu[i] = grafika_P_dla_kraty_2D(spacer_gpu,
				iteracja, width, height);
		}
	}

	__host__ void display_images(ImGuiIO&) {
		grafika* G_cpu = grafiki_iteracji_cpu[pokazywana_grafika];
		grafika* G_gpu = grafiki_iteracji_gpu[pokazywana_grafika];
		ImGui::Text("CPU: t = %lf", spacer_cpu.iteracje_zapamietane[pokazywana_grafika]->czas);
		ImGui::Text("GPU: t = %lf", spacer_gpu.iteracje_zapamietane[pokazywana_grafika]->czas);
		ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, 0.01f, 1.0f);

		plot_grafike_dla_kraty_2D(spacer_cpu, pokazywana_grafika, przestrzen, G_cpu, width, height, skala_obrazu, "spacer cpu");
		ImGui::SameLine();
		plot_grafike_dla_kraty_2D(spacer_gpu, pokazywana_grafika, przestrzen, G_gpu, width, height, skala_obrazu, "spacer gpu");
		
		ImGui::Text("Czas gpu: %ld ms", czas_gpu_ys / 1000UL);
		ImGui::Text("Czas cpu: %ld ms", czas_cpu_ys / 1000UL);

		if (okres_pokazu_slajdow < 0.95f) {
			double czas = glfwGetTime();
			if (czas > (ostatni_czas_odswiezenia + (double)okres_pokazu_slajdow)) {
				ostatni_czas_odswiezenia = czas;
				pokazywana_grafika = (pokazywana_grafika + 1) % grafiki_iteracji_cpu.size();
			}
		}
		else {
			ostatni_czas_odswiezenia = glfwGetTime();
		}
	}

	__host__ void pokaz_okno(ImGuiIO& io) {
		ImGui::Begin(nazwa_okna.c_str());
		ImGui::SliderFloat("Rozmiar grafiki", &skala_obrazu, 0.0f, 10.0f);
		ImGui::SliderInt("Ktora grafika jest pokazywana", &pokazywana_grafika, 0, (int)liczba_zapamietanych_iteracji() - 1);
		ImGui::SliderInt("Liczba iteracji", (int*)&liczba_iteracji, 1, 100000);
		ImGui::SliderInt("Ile pracy na jeden watek", (int*)&ile_pracy_na_jeden_watek, 1, 200);
		ImGui::SliderInt("Max liczba watkow w bloku", (int*)&max_liczba_watkow_w_bloku, 1, 900);
		if(ImGui::Button("Przelicz cpu")){
			przelicz_cpu();
		}
		ImGui::SameLine();
		if (ImGui::Button("Przelicz gpu")) {
			przelicz_gpu();
		}
		display_images(io);
		ImGui::SameLine();
		ImGui::End();
	}

	__host__ ~test_czasow_wykonania_kwantowy() {
		for (auto g : grafiki_iteracji_cpu) {
			delete g;
		}
		for (auto g : grafiki_iteracji_gpu) {
			delete g;
		}
	}
};

struct test_sciezki_spaceru_kwantowy_dyskretny {
	std::vector<grafika*> grafiki_iteracji;
	grafika* celownik = nullptr;
	std::string nazwa_okna;

	uint32_t height = 0;
	uint32_t width = 0;

	double ostatni_czas_odswiezenia = glfwGetTime();
	float okres_pokazu_slajdow = 1.0f;

	int pokazywana_grafika = 0;
	float skala_obrazu = 1.0f;

	std::vector<zesp> katy;
	std::vector<double> katy_norm;
	std::vector<double> prawdopodop;
	std::vector<double> czasy;

	const uint32_t liczba_wierzcholkow_boku = 101;
	const uint32_t liczba_iteracji = 50;
	const uint32_t jak_czesto_zapisac = 1;

	graf przestrzen;
	spacer_losowy<zesp, TMDQ> spacer;

	__host__ test_sciezki_spaceru_kwantowy_dyskretny()
		: nazwa_okna("Test sciezki spaceru kwantowego dyskretny")
		, przestrzen(graf_krata_2D(liczba_wierzcholkow_boku))
		, spacer(spacer_krata_2D<zesp, TMDQ>(liczba_wierzcholkow_boku,
		HxH, I_3, I_2, &przestrzen)) {

		spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(zesp()) / std::sqrt(2.0);
		spacer.iteracjaA[spacer.trwale.wierzcholki[((liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2) + 1].start_wartosci + 2] = jeden(zesp()) / std::sqrt(2.0);

		spacer.zapisz_iteracje();
		for (uint64_t i = 1; i < liczba_iteracji; i++) {
			spacer.iteracja_na_cpu();
			spacer.dokoncz_iteracje(1.0);
			if ((i % jak_czesto_zapisac) == 0) {
				spacer.zapisz_iteracje();
			}
		}
		// permutacja
		TMDQ identycznosc = I_4;
		spacer.trwale.zamien_transformate(0, identycznosc);

		spacer.iteracja_na_cpu();
		spacer.dokoncz_iteracje(1.0);
		spacer.zapisz_iteracje();
		// permutacja

		TMDQ temp = HxH;
		spacer.trwale.zamien_transformate(0, temp);

		spacer.odwroc();


		for (uint64_t i = 1; i < 2 * liczba_iteracji; i++) {
			spacer.iteracja_na_cpu();
			spacer.dokoncz_iteracje(1.0);
			if ((i % jak_czesto_zapisac) == 0) {
				spacer.zapisz_iteracje();
			}
		}

		celownik = new grafika("textures/crosshair.png");
		ASSERT_Z_ERROR_MSG(celownik->texture != 0, "Cos nie tak z textura celownika\n");

		przygotuj_grafiki();
	}

	__host__ uint64_t liczba_zapamietanych_iteracji() {
		return spacer.iteracje_zapamietane.rozmiar;
	}

	__host__ void przygotuj_grafiki() {
		height = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		width = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		ASSERT_Z_ERROR_MSG(height * width == spacer.trwale.liczba_wierzcholkow(), "Tego spaceru nie da sie przedstawic jako kwadrat\n");

		grafiki_iteracji.resize(liczba_zapamietanych_iteracji());
		katy.resize(liczba_zapamietanych_iteracji());
		katy_norm.resize(liczba_zapamietanych_iteracji());
		prawdopodop.resize(liczba_zapamietanych_iteracji());
		czasy.resize(liczba_zapamietanych_iteracji());

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[i]);
			grafiki_iteracji[i] = grafika_P_dla_kraty_2D(spacer, iteracja,
				width, height, &(prawdopodop[i]));
			czasy[i] = iteracja.czas;
		}

		spacer::dane_iteracji<zesp>& iteracja_pierwsza = *(spacer.iteracje_zapamietane[0]);

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[i]);
			katy[i] = dot(iteracja_pierwsza.wartosci, iteracja.wartosci);
			katy_norm[i] = std::sqrt(katy[i].norm());
		}
	}

	__host__ void display_image(ImGuiIO&) {
		grafika* G = grafiki_iteracji[pokazywana_grafika];
		ImGui::Text("t = %lf", spacer.iteracje_zapamietane[pokazywana_grafika]->czas);
		ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, 0.01f, 1.0f);

		plot_grafike_dla_kraty_2D(spacer, pokazywana_grafika, przestrzen, G, width, height, skala_obrazu);

		if (okres_pokazu_slajdow < 0.95f) {
			double czas = glfwGetTime();
			if (czas > (ostatni_czas_odswiezenia + (double)okres_pokazu_slajdow)) {
				ostatni_czas_odswiezenia = czas;
				pokazywana_grafika = (pokazywana_grafika + 1) % grafiki_iteracji.size();
			}
		}
		else {
			ostatni_czas_odswiezenia = glfwGetTime();
		}
	}

	__host__ void pokaz_wykresy(ImGuiIO&) {
		if (ImPlot::BeginPlot("##Dane w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
			ImPlot::PlotInfLines("Vertical pomocnik", &czasy[pokazywana_grafika], 1);
			ImPlot::PlotLine("Prawdopodobienstwa suma", czasy.data(), prawdopodop.data(), (int)liczba_zapamietanych_iteracji());
			ImPlot::PlotLineLepsze("Kat Re", czasy.data(), (double*)katy.data(), (int)liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLineLepsze("Kat Im", czasy.data(), (double*)katy.data() + 1, (int)liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLine("Kat norma", czasy.data(), katy_norm.data(), (int)liczba_zapamietanych_iteracji());
			ImPlot::EndPlot();
		}
	}

	__host__ void pokaz_okno(ImGuiIO& io) {
		ImGui::Begin(nazwa_okna.c_str());
		ImGui::SliderFloat("Rozmiar grafiki", &skala_obrazu, 0.0f, 10.0f);
		ImGui::SliderInt("Ktora grafika jest pokazywana", &pokazywana_grafika, 0, (int)liczba_zapamietanych_iteracji() - 1);
		display_image(io);
		ImGui::SameLine();
		pokaz_wykresy(io);
		ImGui::End();
	}

	__host__ ~test_sciezki_spaceru_kwantowy_dyskretny() {
		delete celownik;
		for (auto g : grafiki_iteracji) {
			delete g;
		}
	}
};



