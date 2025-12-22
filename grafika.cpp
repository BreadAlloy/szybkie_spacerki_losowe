#include "grafika.h"
#include "spacer_losowy.h"
#include "transformaty_wyspecializowane.h"

template <typename towar, typename transformata>
__host__ grafika* grafika_P_dla_kraty_2D(spacer_losowy<towar, transformata>& spacer, spacer::dane_iteracji<towar>& iteracja, uint32_t width, uint32_t height, double* suma_ptr){
	// nie sprawdzam czy iteracja nale퓓 do spaceru.
	// potem trzeba zrobic delete grafika*

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
	}

	for (uint64_t j = 0; j < spacer.trwale.liczba_wierzcholkow(); j++) {
		uint8_t jasnosc_znormalizowana = (uint8_t)((*(float*)((G->data) + 4 * j)) / max);

		(G->data)[4 * j + 0] = jasnosc_znormalizowana;  // R
		(G->data)[4 * j + 1] = jasnosc_znormalizowana;  // G
		(G->data)[4 * j + 2] = jasnosc_znormalizowana;	// B
		(G->data)[4 * j + 3] = 0xFF;
	}

	if(suma_ptr != nullptr) *suma_ptr = prawdopodobienstwo_suma;
	G->LoadTextureFromMemory();
	return G;
}

template __host__ grafika* grafika_P_dla_kraty_2D(spacer_losowy<double, TMDK>& spacer,
				 spacer::dane_iteracji<double>& iteracja, uint32_t width, uint32_t height, double* suma_ptr);

template __host__ grafika* grafika_P_dla_kraty_2D(spacer_losowy<zesp, TMDQ>& spacer,
			     spacer::dane_iteracji<zesp>& iteracja, uint32_t width, uint32_t height, double* suma_ptr);

template __host__ grafika* grafika_P_dla_kraty_2D(spacer_losowy<zesp, TMCQ>& spacer,
				 spacer::dane_iteracji<zesp>& iteracja, uint32_t width, uint32_t height, double* suma_ptr);



template <typename towar, typename transformata>
__host__ void plot_grafike_dla_kraty_2D(spacer_losowy<towar, transformata>& spacer, uint64_t pokazywana_grafika, graf& przestrzen, grafika* G, uint32_t width, uint32_t height, float skala_obrazu, std::string nazwa_wykresu){
	// Nie sprawdza czy grafika nale퓓 do tej iteracji

	ImVec2 bmin(0.0, 0.0);
	ImVec2 bmax((float)height, (float)width);
	ImVec2 uv0(0.0, 0.0);
	ImVec2 uv1(1.0, -1.0); // bo tak tworze osie przy tworzeniu grafu
	if (ImPlot::BeginPlot(nazwa_wykresu.c_str(), ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f), ImPlotFlags_Equal)) {
		ImPlot::PlotImage("Iteracja w spacerze", (ImTextureID)(intptr_t)(G->texture), bmin, bmax, uv0, uv1);
		if (ImPlot::IsPlotHovered() && ImGui::GetIO().KeyCtrl) {
			ImPlotPoint pt = ImPlot::GetPlotMousePos();
			uint64_t i = (uint64_t)(pt.x);
			uint64_t j = (uint64_t)(pt.y);

			if (i < width && j < height && ImGui::BeginItemTooltip()) {
				spacer::wierzcholek& w = spacer.trwale.wierzcholki[i + j * width];
				spacer::dane_iteracji<towar>& iteracja = *(spacer.iteracje_zapamietane[pokazywana_grafika]);
				ImGui::Text("Wierzcholek o indeksie: %d\n", i + j * width);
				ImGui::Text("Indeksy wartosci %ld - %ld\n", w.start_wartosci, w.start_wartosci+w.liczba_kierunkow);
				ImGui::Text("Szczegoly wierzcholka: %s", przestrzen.wierzcholki[i + j * width].opis.c_str());
				pokaz_transformate(spacer.trwale.transformaty[w.transformer]);
				pokaz_stan(estetyczny_wektor<towar>(&(iteracja[w.start_wartosci]), w.liczba_kierunkow));
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
}

template __host__ void plot_grafike_dla_kraty_2D(spacer_losowy<double, TMDK>& spacer, uint64_t pokazywana_grafika, graf& przestrzen, grafika* G, uint32_t width, uint32_t height, float skala_obrazu, std::string nazwa_wykresu);

template __host__ void plot_grafike_dla_kraty_2D(spacer_losowy<zesp, TMDQ>& spacer, uint64_t pokazywana_grafika, graf& przestrzen, grafika* G, uint32_t width, uint32_t height, float skala_obrazu, std::string nazwa_wykresu);

template __host__ void plot_grafike_dla_kraty_2D(spacer_losowy<zesp, TMCQ>& spacer, uint64_t pokazywana_grafika, graf& przestrzen, grafika* G, uint32_t width, uint32_t height, float skala_obrazu, std::string nazwa_wykresu);

template <typename towar, typename transformata>
__host__ std::vector<grafika*> grafiki_P_kierunkow_dla_kraty_2D(spacer_losowy<towar, transformata>& spacer, spacer::dane_iteracji<towar>& iteracja, uint32_t width, uint32_t height) {
	// nie sprawdzam czy iteracja nale퓓 do spaceru.
	// potem trzeba zrobic delete grafika*
	float* data = (float*)malloc(sizeof(float) * 2 * width * 2 * height);

	double max = 0.0;
	double max_0 = 0.0;
	double max_1 = 0.0;
	double max_2 = 0.0;
	double max_3 = 0.0;

	double P_0, P_1, P_2, P_3;
	uint64_t w = 0;
	for (uint64_t i = 0; i < 2 * height; i += 2) {
		for (uint64_t j = 0; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			spacer::wierzcholek& wierzcholek = spacer.trwale.wierzcholki[w];
			P_0 = P(iteracja.wartosci[wierzcholek.start_wartosci + 0]);
			max_0 = MAX(max_0, P_0);

			data[indeks] = (float)(P_0 * 255.0);
			w++;
		}
	}

	w = 0;
	for (uint64_t i = 0; i < 2 * height; i += 2) {
		for (uint64_t j = 1; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			spacer::wierzcholek& wierzcholek = spacer.trwale.wierzcholki[w];
			P_1 = P(iteracja.wartosci[wierzcholek.start_wartosci + 1]);
			max_1 = MAX(max_1, P_1);

			data[indeks] = (float)(P_1 * 255.0);
			w++;
		}
	}

	w = 0;
	for (uint64_t i = 1; i < 2 * height; i += 2) {
		for (uint64_t j = 0; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			spacer::wierzcholek& wierzcholek = spacer.trwale.wierzcholki[w];
			P_2 = P(iteracja.wartosci[wierzcholek.start_wartosci + 2]);
			max_2 = MAX(max_2, P_2);

			data[indeks] = (float)(P_2 * 255.0);
			w++;
		}
	}

	w = 0;
	for (uint64_t i = 1; i < 2 * height; i += 2) {
		for (uint64_t j = 1; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			spacer::wierzcholek& wierzcholek = spacer.trwale.wierzcholki[w];
			P_3 = P(iteracja.wartosci[wierzcholek.start_wartosci + 3]);
			max_3 = MAX(max_3, P_3);

			data[indeks] = (float)(P_3 * 255.0);
			w++;
		}
	}

	max = MAX(max_0, max);
	max = MAX(max_1, max);
	max = MAX(max_2, max);
	max = MAX(max_3, max);

	float jasnosc_znormalizowana;
	for (uint64_t i = 0; i < 2 * height; i += 2) {
		for (uint64_t j = 0; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			float* ptr = data+indeks;
			jasnosc_znormalizowana = *ptr / (float)max;
			//jasnosc_znormalizowana = 255.0f;

			((uint8_t*)ptr)[0] = (uint8_t)(jasnosc_znormalizowana * 108.0f / 255.0f);  // R
			((uint8_t*)ptr)[1] = (uint8_t)(jasnosc_znormalizowana * 255.0f / 255.0f);  // G
			((uint8_t*)ptr)[2] = (uint8_t)(jasnosc_znormalizowana * 66.0f / 255.0f);  // B
			((uint8_t*)ptr)[3] = 0x00;
		}
	}

	for (uint64_t i = 0; i < 2 * height; i += 2) {
		for (uint64_t j = 1; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			float* ptr = data + indeks;
			jasnosc_znormalizowana = *ptr / (float)max;
			//jasnosc_znormalizowana = 255.0f;

			((uint8_t*)ptr)[0] = (uint8_t)(jasnosc_znormalizowana * 255.0f / 255.0f);  // R
			((uint8_t*)ptr)[1] = (uint8_t)(jasnosc_znormalizowana * 77.0f / 255.0f);  // G
			((uint8_t*)ptr)[2] = (uint8_t)(jasnosc_znormalizowana * 223.0f / 255.0f);  // B
			((uint8_t*)ptr)[3] = 0x00;
		}
	}

	for (uint64_t i = 1; i < 2 * height; i += 2) {
		for (uint64_t j = 0; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			float* ptr = data + indeks;
			jasnosc_znormalizowana = *ptr / (float)max;
			//jasnosc_znormalizowana = 255.0f;

			((uint8_t*)ptr)[0] = (uint8_t)(jasnosc_znormalizowana * 255.0f / 255.0f);  // R
			((uint8_t*)ptr)[1] = (uint8_t)(jasnosc_znormalizowana * 187.0f / 255.0f);  // G
			((uint8_t*)ptr)[2] = (uint8_t)(jasnosc_znormalizowana * 77.0f / 255.0f);  // B
			((uint8_t*)ptr)[3] = 0x00;
		}
	}

	for (uint64_t i = 1; i < 2 * height; i += 2) {
		for (uint64_t j = 1; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			float* ptr = data + indeks;
			jasnosc_znormalizowana = *ptr / (float)max;
			//jasnosc_znormalizowana = 255.0f;

			((uint8_t*)ptr)[0] = (uint8_t)(jasnosc_znormalizowana * 66.0f / 255.0f);  // R
			((uint8_t*)ptr)[1] = (uint8_t)(jasnosc_znormalizowana * 164.0f / 255.0f);  // G
			((uint8_t*)ptr)[2] = (uint8_t)(jasnosc_znormalizowana * 255.0f / 255.0f);  // B
			((uint8_t*)ptr)[3] = 0x00;
		}
	}

	std::vector<grafika*> G(4);

	for (uint64_t i = 0; i < 2 * height; i += 2) {
		for (uint64_t j = 0; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			float* ptr = data + indeks;
			((uint8_t*)ptr)[3] = 0xFF;
		}
	}

	G[0] = new grafika(2 * width, 2 * height, (uint8_t*)data);

	for (uint64_t i = 0; i < 2 * height; i += 2) {
		for (uint64_t j = 0; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			float* ptr = data + indeks;
			((uint8_t*)ptr)[3] = 0x00;
		}
	}
	for (uint64_t i = 0; i < 2 * height; i += 2) {
		for (uint64_t j = 1; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			float* ptr = data + indeks;
			((uint8_t*)ptr)[3] = 0xFF;
		}
	}

	G[1] = new grafika(2 * width, 2 * height, (uint8_t*)data);

	for (uint64_t i = 0; i < 2 * height; i += 2) {
		for (uint64_t j = 1; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			float* ptr = data + indeks;
			((uint8_t*)ptr)[3] = 0x00;
		}
	}
	for (uint64_t i = 1; i < 2 * height; i += 2) {
		for (uint64_t j = 0; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			float* ptr = data + indeks;
			((uint8_t*)ptr)[3] = 0xFF;
		}
	}

	G[2] = new grafika(2 * width, 2 * height, (uint8_t*)data);

	for (uint64_t i = 1; i < 2 * height; i += 2) {
		for (uint64_t j = 0; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			float* ptr = data + indeks;
			((uint8_t*)ptr)[3] = 0x00;
		}
	}
	for (uint64_t i = 1; i < 2 * height; i += 2) {
		for (uint64_t j = 1; j < 2 * width; j += 2) {
			uint64_t indeks = j + (i * 2 * width);

			float* ptr = data + indeks;
			((uint8_t*)ptr)[3] = 0xFF;
		}
	}

	G[3] = new grafika(2 * width, 2 * height, (uint8_t*)data);

	free(data);
	return G;
}

template __host__ std::vector<grafika*> grafiki_P_kierunkow_dla_kraty_2D(spacer_losowy<zesp, TMCQ>& spacer, spacer::dane_iteracji<zesp>& iteracja, uint32_t width, uint32_t height);

template <typename towar, typename transformata>
__host__ void plot_spacer_dla_kraty_2D(spacer_losowy<towar, transformata>& spacer, uint64_t pokazywana_grafika, std::vector<grafika*> kierunki, graf& przestrzen, grafika* G, uint32_t width, uint32_t height, float skala_obrazu, std::string nazwa_wykresu) {
	// Nie sprawdza czy grafika nale퓓 do tej iteracji

	ImVec2 bmin(0.0, 0.0);
	ImVec2 bmax((float)height, (float)width);
	ImVec2 uv0(0.0, 0.0);
	ImVec2 uv1(1.0, -1.0); // bo tak tworze osie przy tworzeniu grafu
	if (ImPlot::BeginPlot(nazwa_wykresu.c_str(), ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f), ImPlotFlags_Equal)) {
		ImPlot::PlotImage("P", (ImTextureID)(intptr_t)(G->texture), bmin, bmax, uv0, uv1);
		ImPlot::PlotImage("P_0", (ImTextureID)(intptr_t)(kierunki[0]->texture), bmin, bmax, uv0, uv1);
		ImPlot::PlotImage("P_1", (ImTextureID)(intptr_t)(kierunki[1]->texture), bmin, bmax, uv0, uv1);
		ImPlot::PlotImage("P_2", (ImTextureID)(intptr_t)(kierunki[2]->texture), bmin, bmax, uv0, uv1);
		ImPlot::PlotImage("P_3", (ImTextureID)(intptr_t)(kierunki[3]->texture), bmin, bmax, uv0, uv1);
		if (ImPlot::IsPlotHovered() && ImGui::GetIO().KeyCtrl) {
			ImPlotPoint pt = ImPlot::GetPlotMousePos();
			uint64_t i = (uint64_t)(pt.x);
			uint64_t j = (uint64_t)(pt.y);

			if (i < width && j < height && ImGui::BeginItemTooltip()) {
				spacer::wierzcholek& w = spacer.trwale.wierzcholki[i + j * width];
				spacer::dane_iteracji<towar>& iteracja = *(spacer.iteracje_zapamietane[pokazywana_grafika]);
				ImGui::Text("Wierzcholek o indeksie: %d\n", i + j * width);
				ImGui::Text("Indeksy wartosci %ld - %ld\n", w.start_wartosci, w.start_wartosci + w.liczba_kierunkow);
				ImGui::Text("Szczegoly wierzcholka: %s", przestrzen.wierzcholki[i + j * width].opis.c_str());
				pokaz_transformate(spacer.trwale.transformaty[w.transformer]);
				pokaz_stan(estetyczny_wektor<towar>(&(iteracja[w.start_wartosci]), w.liczba_kierunkow));
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
}

template __host__ void plot_spacer_dla_kraty_2D(spacer_losowy<zesp, TMCQ>& spacer, uint64_t pokazywana_grafika, std::vector<grafika*> kierunki, graf& przestrzen, grafika* G, uint32_t width, uint32_t height, float skala_obrazu, std::string nazwa_wykresu);

glm::vec3 kolor0(108.0f / 255.0f, 255.0f / 255.0f, 66.0f  / 255.0f);
glm::vec3 kolor1(255.0f / 255.0f, 77.0f  / 255.0f, 223.0f / 255.0f);
glm::vec3 kolor2(255.0f / 255.0f, 187.0f / 255.0f, 77.0f  / 255.0f);
glm::vec3 kolor3(66.0f  / 255.0f, 164.0f / 255.0f, 255.0f / 255.0f);

template <typename towar, typename transformata>
__host__ grafika* grafika_P_kierunkow_dla_kraty_2D(spacer_losowy<towar, transformata>& spacer, spacer::dane_iteracji<towar>& iteracja, uint32_t width, uint32_t height, double* suma_ptr) {
	// nie sprawdzam czy iteracja nale퓓 do spaceru.
	// potem trzeba zrobic delete grafika*

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
		max = MAX(max, prawdopodobienstwo);
	}

	float normalizator = (float)(255.0 / max);
	for (uint64_t j = 0; j < spacer.trwale.liczba_wierzcholkow(); j++) {
		spacer::wierzcholek& wierzcholek = spacer.trwale.wierzcholki[j];

		glm::vec3 kolor(0.0f, 0.0f, 0.0f);
		kolor += normalizator * (float)P(iteracja.wartosci[wierzcholek.start_wartosci + 0]) * kolor0;
		kolor += normalizator * (float)P(iteracja.wartosci[wierzcholek.start_wartosci + 1]) * kolor1;
		kolor += normalizator * (float)P(iteracja.wartosci[wierzcholek.start_wartosci + 2]) * kolor2;
		kolor += normalizator * (float)P(iteracja.wartosci[wierzcholek.start_wartosci + 3]) * kolor3;

		(G->data)[4 * j + 0] = (uint8_t)kolor.x;  // R
		(G->data)[4 * j + 1] = (uint8_t)kolor.y;  // G
		(G->data)[4 * j + 2] = (uint8_t)kolor.z;  // B
		(G->data)[4 * j + 3] = (uint8_t)0xFF;
	}

	if (suma_ptr != nullptr) *suma_ptr = prawdopodobienstwo_suma;
	G->LoadTextureFromMemory();
	return G;
}

template __host__ grafika* grafika_P_kierunkow_dla_kraty_2D(spacer_losowy<zesp, TMCQ>& spacer, spacer::dane_iteracji<zesp>& iteracja, uint32_t width, uint32_t height, double* suma_ptr);