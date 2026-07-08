#pragma once

#include "spacer_losowy.h"

#include "grafika.h"

#include "transformaty_wyspecializowane.h"

#include "imgui_spacer.h"

#include "definicje_typowych_macierzy.h"

#include "alg_liniowa.h"

#include "screenshot.h"

namespace spojrzenie_warstwy {
	struct krata_TMSQ {
		std::string nazwa_okna;

		std::vector<std::vector<grafika*>> grafiki_warstw;

		int ogladany_czas = 0;
		float skala_obrazu = 1.0f;
		float wzmocnienie = 1.0f;

		std::vector<prob_t> prawdopodop_suma;
		std::vector<std::vector<prob_t>> prawdopodop;
		std::vector<fp_t> czasy;

		static constexpr uint8_t liczba_warstw = 3;

		static constexpr uint32_t skalar_instancji = 1;
		static constexpr uint32_t liczba_wierzcholkow_boku = 100 * skalar_instancji;// + 1;

		static constexpr uint32_t pozycja_poczatkowa_1 = 5 * skalar_instancji;
		static constexpr uint32_t pozycja_poczatkowa_2 = 95 * skalar_instancji;

		static constexpr uint32_t liczba_iteracji = 20000 * skalar_instancji;
		//static constexpr uint32_t liczba_iteracji = 1000;
		static constexpr uint32_t jak_czesto_zapisac = 100 * skalar_instancji; //* skalar_instancji;
		//static constexpr uint32_t jak_czesto_zapisac = liczba_iteracji - 1; //* skalar_instancji;

		TMSQ transformata = tensor(HxH, I_3);

		graf przestrzen;
		graf przestrzen_warstwowa;
		indekser_warstwowy indekser;
		spacer_losowy<zesp, TMSQ> spacer;

		__host__ krata_TMSQ()
			: nazwa_okna("Spojrzenie warstwy krata TMSQ")
			, przestrzen(graf_krata_2D_cykl(liczba_wierzcholkow_boku))
			, spacer(graf())
		{
			std::tie(przestrzen_warstwowa, indekser) = przestrzen.warstwowy(liczba_warstw);
			
			spacer = spacer_losowy<zesp, TMSQ>(przestrzen_warstwowa);
			spacer::uklad_transformat<TMSQ> transformaty = uklad_transformat_wszystko_to_samo<TMSQ>(spacer.trwale.liczba_wierzcholkow(), transformata);
			spacer.trwale.dodaj_transformaty(transformaty);
			spacer.trwale.przygotuj_znajdywacz_wierzcholka();
			spacer.przygotuj_pierwsza_iteracje();
			
			reset_spacery();

			spacer.czy_gotowy();
			
			grafiki_warstw.resize(liczba_warstw);
			prawdopodop.resize(liczba_warstw);

			policz_gpu();
			przygotuj_grafiki();
		}

#if 0
		void policz_cpu() {
			printf("CPU start\n");
			for (uint32_t i = 0; i < liczba_iteracji; i++) {
				spacer_1_czastka1.iteracja_na_cpu();
				if (i % jak_czesto_zapisac == 0) {
					spacer_1_czastka1.zapisz_iteracje();
				}
				spacer_1_czastka1.dokoncz_iteracje(dt);
			}

			printf("CPU koniec\n");
		}
#endif

		void policz_gpu() {
			printf("CUDA start\n");

			CZAS_INIT
			CZAS_START

			spacer.zbuduj_na_cuda();
			podzielone_iteracje_na_gpu<zesp, TMSQ>(spacer, dt, liczba_iteracji, 70, 300, jak_czesto_zapisac);
			spacer.cuda_przynies();
			spacer.zburz_na_cuda();

			CZAS_STOP
			printf("CUDA koniec\n");
		}

		__host__ uint64_t liczba_zapamietanych_iteracji() {
			return spacer.iteracje_zapamietane.rozmiar;
		}

		__host__ void przygotuj_grafiki(uint64_t rozmiar_przed = std::numeric_limits<uint64_t>::max()) {
			ASSERT_Z_ERROR_MSG(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku ==
				spacer.trwale.liczba_wierzcholkow(), "Tego spaceru nie da sie przedstawic jako kwadrat\n");

			if (rozmiar_przed == std::numeric_limits<uint64_t>::max()) rozmiar_przed = grafiki_warstw[0].size();
			
			for(auto& grafiki_warstwy : grafiki_warstw){
				grafiki_warstwy.resize(liczba_zapamietanych_iteracji());
			}

			for (auto& probs : prawdopodop) {
				probs.resize(liczba_zapamietanych_iteracji());
			}
	
			prawdopodop_suma.resize(liczba_zapamietanych_iteracji());
			czasy.resize(liczba_zapamietanych_iteracji());

			for (uint64_t i = rozmiar_przed; i < liczba_zapamietanych_iteracji(); i++) {
				spacer::dane_iteracji<zesp>* iteracja = spacer.iteracje_zapamietane[i];
				
				prawdopodop_suma[i] = FP_ZERO;

				for(uint8_t k = 0; k < liczba_warstw; k++){
					grafiki_warstw[k][i] = grafika_P_kierunkow_dla_warstwy_kraty_2D(spacer, (iteracja->wartosci),
						liczba_wierzcholkow_boku, liczba_wierzcholkow_boku, k, indekser, &(prawdopodop[k][i]), wzmocnienie);

					prawdopodop_suma[i] += prawdopodop[k][i];
				}

				czasy[i] = iteracja->czas;
			}
		}

		__host__ void pokaz_wykresy() {
			if (ImPlot::BeginPlot("##Dane w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
				ImPlot::PlotInfLines("Vertical pomocnik", &czasy[ogladany_czas], 1);
				for(uint8_t k = 0; k < liczba_warstw; k++){
					ImPlot::PlotLine(("Prawdopodobienstwa suma warstwa:" + std::to_string(k)).c_str()
						, czasy.data(), prawdopodop[k].data(), (int)liczba_zapamietanych_iteracji());
				}

				ImPlot::PlotLine("Prawdopodobienstwa suma suma"
					, czasy.data(), prawdopodop_suma.data(), (int)liczba_zapamietanych_iteracji());

				ImPlot::EndPlot();
			}
		}

		__host__ void pokaz_spacery() {
			ImGui::Begin(("Wykresy " + nazwa_okna).c_str());
			ImGui::Text("t = %lf", czasy[ogladany_czas]);
			for(uint8_t k = 0; k < liczba_warstw; k++){

				plot_spacer_dla_kraty_2D(spacer, ogladany_czas, przestrzen_warstwowa,
					grafiki_warstw[k][ogladany_czas], liczba_wierzcholkow_boku, liczba_wierzcholkow_boku,
					skala_obrazu, "Spacer warstwa:" + std::to_string(k));

				if( (k + 1) != liczba_warstw ) ImGui::SameLine();
			}

			pokaz_wykresy();
			ImGui::SameLine();
			pokaz_transformate(transformata);

			ImGui::End();
		}

		__host__ void reset_spacery() {
			spacer.reset();

			spacer.iteracjaA[spacer.trwale.wierzcholki
				[(liczba_wierzcholkow_boku / 2) + (liczba_wierzcholkow_boku / 2) * liczba_wierzcholkow_boku].start_wartosci]
				 = zesp(1.0, 0.0);

			spacer.iteracjaA[spacer.trwale.wierzcholki
				[(liczba_wierzcholkow_boku / 3) + (liczba_wierzcholkow_boku / 3) * liczba_wierzcholkow_boku].start_wartosci + 4]
				= zesp(1.0, 0.0);

			spacer.iteracjaA[spacer.trwale.wierzcholki
				[(2 * liczba_wierzcholkow_boku / 3) + (2 * liczba_wierzcholkow_boku / 3) * liczba_wierzcholkow_boku].start_wartosci + 8]
				= zesp(1.0, 0.0);
		}

		__host__ void reset_grafiki() {
			for (uint8_t k = 0; k < liczba_warstw; k++) {
				for (auto g : grafiki_warstw[k]) delete g;
				grafiki_warstw[k].resize(0);
			}
		}

		__host__ void kolejne_transformaty() {
			ogladany_czas = 0;

			reset_spacery();
			transformata = losowa_transformata(4 * liczba_warstw);

			spacer.trwale.zamien_transformate(0, transformata);

			policz_gpu();
			reset_grafiki();
			przygotuj_grafiki();

		}

		__host__ ~krata_TMSQ() {
			for(uint8_t k = 0; k < liczba_warstw; k++){
				for (auto g : grafiki_warstw[k]) delete g;
			}
		}
	};

	struct przegladacz_instancji : krata_TMSQ {
		static constexpr bool zapisz = false;
		static constexpr bool daj_nowa_instancje = false;
		static constexpr int przeskok_czasu = 1; // ju¿ po zapisaniu co niektórej iteracji

		double ostatni_czas_odswiezenia = glfwGetTime();
		float okres_pokazu_slajdow = 1.0f;

		// folder musi istnieæ
		std::string folder = "screenshots";
		std::vector<char> filename;

		screenshot aparat;
		grafika zdjecie;

		static constexpr uint32_t screenshot_scale = 120;
		static constexpr uint32_t screenshot_width = 15 * screenshot_scale;
		static constexpr uint32_t screenshot_height = 11 * screenshot_scale;

		przegladacz_instancji()
			: aparat(screenshot_width, screenshot_height), zdjecie(screenshot_width, screenshot_height)
		{
			nowy_screenshot();
			filename.resize(100);
		}

		void tick() {
			if (okres_pokazu_slajdow < 0.95f) {
				double czas = glfwGetTime();
				if (czas > (ostatni_czas_odswiezenia + (double)okres_pokazu_slajdow)) {
					if (nastepny_obraz() && daj_nowa_instancje) {
						kolejne_transformaty();
					}
					ostatni_czas_odswiezenia = glfwGetTime();
				}
			}
			else {
				ostatni_czas_odswiezenia = glfwGetTime();
			}

			pokaz_kontrolki();

			ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
			ImGui::SetNextWindowSize(ImVec2(6.0f + (float)screenshot_width, 30.0f + (float)screenshot_height));
			pokaz_spacery();
		}

		__host__ void nowy_screenshot() {
			aparat.screen_cap(zdjecie.data);
			zdjecie.LoadTextureFromMemory(false);
		}

		__host__ void zapisz_screenshot(std::string nazwa) {
			zdjecie.SaveToFile(folder + "//" + nazwa + ".bmp");
		}

		void pokaz_kontrolki() {
			ImGui::Begin(("Kontrolki: " + nazwa_okna).c_str());
			ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, 0.01f, 1.0f);
			ImGui::SliderFloat("Rozmiar grafiki", &skala_obrazu, 0.0f, 10.0f);
			ImGui::SliderInt("Ktora grafika jest pokazywana", &ogladany_czas, 0, (int)liczba_zapamietanych_iteracji() - 1);
			if (ImGui::Button("Policz wiecej")) {
				policz_gpu();
				przygotuj_grafiki();
			}

			ImGui::SliderFloat("Wzmocnienie", &wzmocnienie, 1.0f, 10.0f);
			if (ImGui::Button("Policz grafiki")) {
				przygotuj_grafiki(0);
			}

			ImGui::InputText("Nazwa screenshota do zapisania", filename.data(), filename.size());
			if (ImGui::Button("Zapisz screenshot")) {
				zapisz_screenshot(filename.data());
			}

			if (ImGui::Button("Nowy screenshot")) {
				nowy_screenshot();
			}

			if (ImGui::Button("Kolejne transformaty")) {
				kolejne_transformaty();
			}

			zdjecie.plot_image("Preview screenshota");
			ImGui::End();
		}

		bool nastepny_obraz() {
			ogladany_czas += przeskok_czasu;
			if (ogladany_czas >= liczba_zapamietanych_iteracji()) {
				ogladany_czas = 0;
				return true;
			}
			return false;
		}

	};



}
