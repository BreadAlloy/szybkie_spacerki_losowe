#pragma once

#undef IM_ASSERT
#define IM_ASSERT(_EXPR)  lepszy_assert(_EXPR)
#include "imgui_internal.h"

#include "spacer_losowy.h"

#include "grafika.h"

#include "transformaty_wyspecializowane.h"

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

	__host__ void display_image(ImGuiIO& io) {
		grafika* G = grafiki_iteracji[pokazywana_grafika];
		ImGui::Text("t = %lf", spacer.iteracje_zapamietane[pokazywana_grafika]->czas);
		ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, 0.01f, 1.0f);

		ImVec2 bmin(0.0, 0.0);
		ImVec2 bmax((float)height, (float)width);
		ImVec2 uv0(0.0, 0.0);
		ImVec2 uv1(1.0, -1.0); // bo tak tworze osie przy tworzeniu grafu
		if (ImPlot::BeginPlot("##iteracje w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f), ImPlotFlags_Equal)) {
			ImPlot::PlotImage("Iteracja w spacerze", (ImTextureID)(intptr_t)(G->texture), bmin, bmax, uv0, uv1);
			if (ImPlot::IsPlotHovered() && ImGui::GetIO().KeyCtrl) {
				ImPlotPoint pt = ImPlot::GetPlotMousePos();
				uint64_t i = (uint64_t)(pt.x);
				uint64_t j = (uint64_t)(pt.y);

				if (i < width && j < height && ImGui::BeginItemTooltip()) {
					spacer::wierzcholek& w = spacer.trwale.wierzcholki[i + j * width];
					spacer::dane_iteracji<double>& iteracja = *(spacer.iteracje_zapamietane[pokazywana_grafika]);
					ImGui::Text("Szczegoly wierzcholka: %s", przestrzen.wierzcholki[i + j * width].opis.c_str());
					pokaz_transformate(spacer.trwale.transformaty[w.transformer]);
					pokaz_stan(estetyczny_wektor<double>(&(iteracja[w.start_wartosci]), w.liczba_kierunkow));
					ImGui::EndTooltip();
				}

				double temp_x = (double)i;
				double temp_y = (double)j;
				double vals_x[4] = { temp_x - 1.0, temp_x, temp_x + 1.0, temp_x + 2.0 };
				double vals_y[4] = { temp_y - 1.0, temp_y, temp_y + 1.0, temp_y + 2.0 };

				ImPlot::PlotInfLines("Vertical pomocnik", vals_x, 4);
				ImPlot::PlotInfLines("Horizontal pomocnik", vals_y, 4, ImPlotInfLinesFlags_Horizontal);

			}
			ImPlot::EndPlot();
		}

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

	zesp macierz3x3[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
	zesp macierz4x4[16] = { 0.5,  0.5,  0.5,  0.5,
							0.5, -0.5,  0.5, -0.5,
							0.5,  0.5, -0.5, -0.5,
							0.5, -0.5, -0.5,  0.5 };

	const uint32_t liczba_wierzcholkow_boku = 21;
	const uint32_t liczba_iteracji = 100;
	const uint32_t jak_czesto_zapisac = 1;

	graf przestrzen;
	spacer_losowy<zesp, TMDQ> spacer;

	__host__ test_spaceru_kwantowy_dyskretny()
		: nazwa_okna("Test spaceru kwantowego dyskretny")
		, przestrzen(graf_krata_2D(liczba_wierzcholkow_boku))
		, spacer(spacer_krata_2D<zesp, TMDQ>(liczba_wierzcholkow_boku, transformata_macierz<zesp>(4, macierz4x4), transformata_macierz<zesp>(3, macierz3x3), transformata_macierz<zesp>(1.0, 0.0, 0.0, 1.0), &przestrzen)) {

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
			grafika* G = new grafika(width, height);
			double max = 0.0;
			double prawdopodobienstwo_suma = 0.0;

			for (uint64_t j = 0; j < spacer.trwale.liczba_wierzcholkow(); j++) {
				double prawdopodobienstwo = 0.0;
				spacer::wierzcholek& wierzcholek = spacer.trwale.wierzcholki[j];

				for (uint8_t k = 0; k < wierzcholek.liczba_kierunkow; k++) {
					prawdopodobienstwo += P(iteracja.wartosci[wierzcholek.start_wartosci + k]);
				}

				prawdopodobienstwo_suma += prawdopodobienstwo;
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
			grafiki_iteracji[i] = G;
			czasy[i] = iteracja.czas;
			prawdopodop[i] = prawdopodobienstwo_suma;
		}

		spacer::dane_iteracji<zesp>& iteracja_pierwsza = *(spacer.iteracje_zapamietane[0]);

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[i]);
			katy[i] = dot(iteracja_pierwsza.wartosci, iteracja.wartosci);
			katy_norm[i] = std::sqrt(katy[i].norm());
		}
	}

	__host__ void display_image(ImGuiIO& io) {
		grafika* G = grafiki_iteracji[pokazywana_grafika];
		ImGui::Text("t = %lf", spacer.iteracje_zapamietane[pokazywana_grafika]->czas);
		ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, 0.01f, 1.0f);

		ImVec2 bmin(0.0, 0.0);
		ImVec2 bmax((float)height, (float)width);
		ImVec2 uv0(0.0, 0.0);
		ImVec2 uv1(1.0, -1.0); // bo tak tworze osie przy tworzeniu grafu
		if (ImPlot::BeginPlot("##iteracje w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f), ImPlotFlags_Equal)) {
			ImPlot::PlotImage("Iteracja w spacerze", (ImTextureID)(intptr_t)(G->texture), bmin, bmax, uv0, uv1);
			if (ImPlot::IsPlotHovered() && ImGui::GetIO().KeyCtrl) {
				ImPlotPoint pt = ImPlot::GetPlotMousePos();
				uint64_t i = (uint64_t)(pt.x);
				uint64_t j = (uint64_t)(pt.y);

				if (i < width && j < height && ImGui::BeginItemTooltip()) {
					spacer::wierzcholek& w = spacer.trwale.wierzcholki[i + j * width];
					spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[pokazywana_grafika]);
					ImGui::Text("Szczegoly wierzcholka: %s", przestrzen.wierzcholki[i + j * width].opis.c_str());
					pokaz_transformate(spacer.trwale.transformaty[w.transformer]);
					pokaz_stan(estetyczny_wektor<zesp>(&(iteracja[w.start_wartosci]), w.liczba_kierunkow));
					ImGui::EndTooltip();
				}

				double temp_x = (double)i;
				double temp_y = (double)j;
				double vals_x[4] = { temp_x - 1.0, temp_x, temp_x + 1.0, temp_x + 2.0 };
				double vals_y[4] = { temp_y - 1.0, temp_y, temp_y + 1.0, temp_y + 2.0 };

				ImPlot::PlotInfLines("Vertical pomocnik", vals_x, 4);
				ImPlot::PlotInfLines("Horizontal pomocnik", vals_y, 4, ImPlotInfLinesFlags_Horizontal);

			}
			ImPlot::EndPlot();
		}

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

	__host__ void pokaz_wykresy(ImGuiIO& io){
		if (ImPlot::BeginPlot("##Dane w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
			ImPlot::PlotInfLines("Vertical pomocnik", &czasy[pokazywana_grafika], 1);
			ImPlot::PlotLine("Prawdopodobienstwa suma", czasy.data(), prawdopodop.data(), liczba_zapamietanych_iteracji());
			ImPlot::PlotLineLepsze("Kat Re", czasy.data(), (double*)katy.data(), liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLineLepsze("Kat Im", czasy.data(), (double*)katy.data()+1, liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLine("Kat norma", czasy.data(), katy_norm.data(), liczba_zapamietanych_iteracji());
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

	zesp macierz3x3[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
	zesp macierz4x4[16] = { 0.5,  0.5,  0.5,  0.5,
							0.5, -0.5,  0.5, -0.5,
							0.5,  0.5, -0.5, -0.5,
							0.5, -0.5, -0.5,  0.5 };

	const uint32_t liczba_wierzcholkow_boku = 101;
	const uint32_t liczba_iteracji = 1000;
	const uint32_t jak_czesto_zapisac = 5;

	graf przestrzen;
	spacer_losowy<zesp, TMDQ> spacer;

	__host__ test_spaceru_kwantowy_dyskretny_gpu()
		: nazwa_okna("Test spaceru kwantowego dyskretny GPU")
		, przestrzen(graf_krata_2D(liczba_wierzcholkow_boku))
		, spacer(spacer_krata_2D<zesp, TMDQ>(liczba_wierzcholkow_boku, transformata_macierz<zesp>(4, macierz4x4), transformata_macierz<zesp>(3, macierz3x3), transformata_macierz<zesp>(1.0, 0.0, 0.0, 1.0), &przestrzen)) {

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
			grafika* G = new grafika(width, height);
			double max = 0.0;
			double prawdopodobienstwo_suma = 0.0;

			for (uint64_t j = 0; j < spacer.trwale.liczba_wierzcholkow(); j++) {
				double prawdopodobienstwo = 0.0;
				spacer::wierzcholek& wierzcholek = spacer.trwale.wierzcholki[j];

				for (uint8_t k = 0; k < wierzcholek.liczba_kierunkow; k++) {
					prawdopodobienstwo += P(iteracja.wartosci[wierzcholek.start_wartosci + k]);
				}

				prawdopodobienstwo_suma += prawdopodobienstwo;
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
			grafiki_iteracji[i] = G;
			czasy[i] = iteracja.czas;
			prawdopodop[i] = prawdopodobienstwo_suma;
		}

		spacer::dane_iteracji<zesp>& iteracja_pierwsza = *(spacer.iteracje_zapamietane[0]);

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[i]);
			katy[i] = dot(iteracja_pierwsza.wartosci, iteracja.wartosci);
			katy_norm[i] = std::sqrt(katy[i].norm());
		}
	}

	__host__ void display_image(ImGuiIO& io) {
		grafika* G = grafiki_iteracji[pokazywana_grafika];
		ImGui::Text("t = %lf", spacer.iteracje_zapamietane[pokazywana_grafika]->czas);
		ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, 0.01f, 1.0f);

		ImVec2 bmin(0.0, 0.0);
		ImVec2 bmax((float)height, (float)width);
		ImVec2 uv0(0.0, 0.0);
		ImVec2 uv1(1.0, -1.0); // bo tak tworze osie przy tworzeniu grafu
		if (ImPlot::BeginPlot("##iteracje w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f), ImPlotFlags_Equal)) {
			ImPlot::PlotImage("Iteracja w spacerze", (ImTextureID)(intptr_t)(G->texture), bmin, bmax, uv0, uv1);
			if (ImPlot::IsPlotHovered() && ImGui::GetIO().KeyCtrl) {
				ImPlotPoint pt = ImPlot::GetPlotMousePos();
				uint64_t i = (uint64_t)(pt.x);
				uint64_t j = (uint64_t)(pt.y);

				if (i < width && j < height && ImGui::BeginItemTooltip()) {
					spacer::wierzcholek& w = spacer.trwale.wierzcholki[i + j * width];
					spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[pokazywana_grafika]);
					ImGui::Text("Szczegoly wierzcholka: %s", przestrzen.wierzcholki[i + j * width].opis.c_str());
					pokaz_transformate(spacer.trwale.transformaty[w.transformer]);
					pokaz_stan(estetyczny_wektor<zesp>(&(iteracja[w.start_wartosci]), w.liczba_kierunkow));
					ImGui::EndTooltip();
				}

				double temp_x = (double)i;
				double temp_y = (double)j;
				double vals_x[4] = { temp_x - 1.0, temp_x, temp_x + 1.0, temp_x + 2.0 };
				double vals_y[4] = { temp_y - 1.0, temp_y, temp_y + 1.0, temp_y + 2.0 };

				ImPlot::PlotInfLines("Vertical pomocnik", vals_x, 4);
				ImPlot::PlotInfLines("Horizontal pomocnik", vals_y, 4, ImPlotInfLinesFlags_Horizontal);

			}
			ImPlot::EndPlot();
		}

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

	__host__ void pokaz_wykresy(ImGuiIO& io) {
		if (ImPlot::BeginPlot("##Dane w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
			ImPlot::PlotInfLines("Vertical pomocnik", &czasy[pokazywana_grafika], 1);
			ImPlot::PlotLine("Prawdopodobienstwa suma", czasy.data(), prawdopodop.data(), liczba_zapamietanych_iteracji());
			ImPlot::PlotLineLepsze("Kat Re", czasy.data(), (double*)katy.data(), liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLineLepsze("Kat Im", czasy.data(), (double*)katy.data() + 1, liczba_zapamietanych_iteracji(), 0, 0, sizeof(zesp));
			ImPlot::PlotLine("Kat norma", czasy.data(), katy_norm.data(), liczba_zapamietanych_iteracji());
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


