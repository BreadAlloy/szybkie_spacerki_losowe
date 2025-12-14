#include "grafika.h"
#include "spacer_losowy.h"
#include "transformaty_wyspecializowane.h"

template <typename towar, typename transformata>
__host__ grafika* grafika_P_dla_kraty_2D(spacer_losowy<towar, transformata>& spacer, spacer::dane_iteracji<towar>& iteracja, uint32_t width, uint32_t height, double* suma_ptr){
	// nie sprawdzam czy iteracja nale¿y do spaceru.
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
	// Nie sprawdza czy grafika nale¿y do tej iteracji

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