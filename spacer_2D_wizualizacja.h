#pragma once

#undef IM_ASSERT
#define IM_ASSERT(_EXPR)  lepszy_assert(_EXPR)
#include "imgui_internal.h"

#include "spacer_losowy.h"

#include "grafika.h"

template<typename towar, typename transformata>
struct spacer_2D_wizalizacja{
	spacer_losowy<towar, transformata> spacer;
	graf przestrzen;

	std::vector<grafika*> grafiki_iteracji;
	grafika* celownik = nullptr;
	std::string nazwa_okna;

	uint32_t height = 0;
	uint32_t width = 0;

	double ostatni_czas_odswiezenia = glfwGetTime();
	float okres_pokazu_slajdow = 1.0f;

	int pokazywana_grafika = 0;
	float skala_obrazu = 1.0f;

	__host__ spacer_2D_wizalizacja(spacer_losowy<towar, transformata>& spacer, graf& przestrzen, std::string nazwa_okna)
	: spacer(spacer), przestrzen(przestrzen), nazwa_okna(nazwa_okna){
		celownik = new grafika("textures/crosshair.png");
		ASSERT_Z_ERROR_MSG(celownik->texture != 0, "Cos nie tak z textura celownika\n");

		przygotuj_grafiki();
	}

	__host__ void przygotuj_grafiki(){
		height = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		width = (uint32_t)std::sqrt(spacer.trwale.liczba_wierzcholkow());
		ASSERT_Z_ERROR_MSG(height * width == spacer.trwale.liczba_wierzcholkow(), "Tego spaceru nie da sie przedstawic jako kwadrat\n");

		for(uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++){
			spacer::dane_iteracji<towar>& iteracja = *(spacer.iteracje_zapamietane[i]);
			grafika* G = new grafika(width, height);
			double max = 0.0;

			for(uint64_t j = 0; j < spacer.trwale.liczba_wierzcholkow(); j++){
				double prawdopodobienstwo = 0.0;
				spacer::wierzcholek& wierzcholek = spacer.trwale.wierzcholki[j];

				for(uint8_t k = 0; k < wierzcholek.liczba_kierunkow; k++){
					prawdopodobienstwo += P(iteracja.wartosci[wierzcholek.start_wartosci + k]);
				}

				if(prawdopodobienstwo > max){
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
				uint8_t jasnosc_znormalizowana = (uint8_t)((*(float*)((G->data) + 4 * j))/max);

				(G->data)[4 * j + 0] = jasnosc_znormalizowana;  // R
				(G->data)[4 * j + 1] = jasnosc_znormalizowana;  // G
				(G->data)[4 * j + 2] = jasnosc_znormalizowana;	// B
				(G->data)[4 * j + 3] = 0xFF;
			}

			G->LoadTextureFromMemory();
			grafiki_iteracji.push_back(G);

		}
	}

	void display_image_lokalny(ImGuiIO& io) {
		grafika* G = grafiki_iteracji[pokazywana_grafika];

		float my_tex_w = (float)G->width * skala_obrazu;
		float my_tex_h = (float)G->height * skala_obrazu;

		static bool use_text_color_for_tint = false;
		ImGui::Text("%.0fx%.0f", my_tex_w, my_tex_h);
		ImVec2 pos = ImGui::GetCursorScreenPos();
		ImVec2 uv_min = ImVec2(0.0f, 0.0f);                 // Top-left
		ImVec2 uv_max = ImVec2(1.0f, 1.0f);                 // Lower-right
		ImVec4 tint_col = use_text_color_for_tint ? ImGui::GetStyleColorVec4(ImGuiCol_Text) : ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // No tint
		ImVec4 border_col = ImGui::GetStyleColorVec4(ImGuiCol_Border);

		ImGui::Text("t = %lf", spacer.iteracje_zapamietane[pokazywana_grafika]->czas);

		ImGui::Image((ImTextureID)(intptr_t)G->texture, ImVec2(G->width * skala_obrazu, G->height * skala_obrazu));

		if (ImGui::BeginItemTooltip())
		{
			float region_sz = 32.0f * 4.0f;
			float region_x = io.MousePos.x - pos.x - region_sz * 0.5f;
			float region_y = io.MousePos.y - pos.y - region_sz * 0.5f;
			uint32_t x = (uint32_t)((io.MousePos.y - pos.y) / skala_obrazu);
			uint32_t y = (uint32_t)((io.MousePos.x - pos.x) / skala_obrazu);
			//wierzcholek badany = (id(x, y));
			//ImGui::Text("t = %d, n = %d", pokazywany.t, G->width * x + y);
			//ImGui::Text(badany.str().c_str());
			float zoom = 4.0f;
			float offset_x = 37.0f + 32.0f * 6.0f;
			float offset_y = offset_x + 32.0f;
			if (region_x < 0.0f) {
				offset_x += region_x * zoom;
				region_x = 0.0f;
			}
			else if (region_x > my_tex_w - region_sz) {
				offset_x += (region_x - (my_tex_w - region_sz)) * zoom;
				region_x = my_tex_w - region_sz;
			}
			if (region_y < 0.0f) {
				offset_y += region_y * zoom;
				region_y = 0.0f;
			}
			else if (region_y > my_tex_h - region_sz) {
				offset_y += (region_y - (my_tex_h - region_sz)) * zoom;
				region_y = my_tex_h - region_sz;
			}
			//ImGui::Text("Min: (%.2f, %.2f)", region_x, region_y);
			//ImGui::Text("Max: (%.2f, %.2f)", region_x + region_sz, region_y + region_sz);
			ImVec2 uv0 = ImVec2((region_x) / my_tex_w, (region_y) / my_tex_h);
			ImVec2 uv1 = ImVec2((region_x + region_sz) / my_tex_w, (region_y + region_sz) / my_tex_h);
			ImGui::Image((ImTextureID)(intptr_t)G->texture, ImVec2(region_sz * zoom, region_sz * zoom), uv0, uv1, tint_col, border_col);
			pos = ImGui::GetWindowPos();
			ImGuiWindow* window = ImGui::GetCurrentWindow();
			window->DrawList->AddImage((ImTextureID)(intptr_t)(celownik->texture), ImVec2(pos.x + offset_x, pos.y + offset_y), ImVec2(pos.x + 75.0f + offset_x, pos.y + 75.0f + offset_y), ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), ImGui::GetColorU32(tint_col));
			//ladna_tabelka(*(badany.transformer));
			ImGui::EndTooltip();
		}
	}

	__host__ void nowy_display_image(ImGuiIO& io){
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

				if (i < width && j < height && ImGui::BeginItemTooltip()){
					spacer::wierzcholek& w = spacer.trwale.wierzcholki[i + j * width];
					spacer::dane_iteracji<towar>& iteracja = *(spacer.iteracje_zapamietane[pokazywana_grafika]);
					ImGui::Text("Szczegoly wierzcholka: %s", przestrzen.wierzcholki[i + j * width].opis.c_str());
					pokaz_transformate(spacer.trwale.transformaty[w.transformer]);
					pokaz_stan(estetyczny_wektor<double>(&(iteracja[w.start_wartosci]), w.liczba_kierunkow));
					ImGui::EndTooltip();
				}

				double temp_x = (double)i;
				double temp_y = (double)j;
				double vals_x[4] = {temp_x-1.0, temp_x, temp_x+1.0, temp_x+2.0};
				double vals_y[4] = {temp_y-1.0, temp_y, temp_y+1.0, temp_y+2.0};
		
				ImPlot::PlotInfLines("Vertical pomocnik", vals_x, 4);
				ImPlot::PlotInfLines("Horizontal pomocnik", vals_y, 4, ImPlotInfLinesFlags_Horizontal);

			}
			ImPlot::EndPlot();
		}

		if(okres_pokazu_slajdow < 0.95f){
			double czas = glfwGetTime();
			if(czas > (ostatni_czas_odswiezenia + (double)okres_pokazu_slajdow)){
				ostatni_czas_odswiezenia = czas;
				pokazywana_grafika = (pokazywana_grafika + 1) % grafiki_iteracji.size();
			}
		} else {
			ostatni_czas_odswiezenia = glfwGetTime();
		}
	}

	__host__ void pokaz_okno(ImGuiIO& io){
		ImGui::Begin(nazwa_okna.c_str());
		ImGui::SliderFloat("Rozmiar grafiki", &skala_obrazu, 0.0f, 10.0f);
		ImGui::SliderInt("Ktora grafika jest pokazywana", &pokazywana_grafika, 0, (int)grafiki_iteracji.size() - 1);
		//display_image_lokalny(io);
		nowy_display_image(io);
		ImGui::End();
	}

	__host__ ~spacer_2D_wizalizacja(){
		delete celownik;
		for(auto g : grafiki_iteracji){
			delete g;
		}
	}







};

