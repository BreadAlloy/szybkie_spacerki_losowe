#pragma once

#include "spacer_losowy.h"

#include "grafika.h"

#include "transformaty_wyspecializowane.h"

#include "imgui_spacer.h"

#include "definicje_typowych_macierzy.h"

#include "alg_liniowa.h"

#include "screenshot.h"

namespace spojrzenie_2_czastki{
struct linia_TMDQ{
	std::string nazwa_okna;

	std::vector<grafika*> grafiki_2_czastki;
	std::vector<grafika*> grafiki_2_czastki_1;
	std::vector<grafika*> grafiki_2_czastki_2;	

	std::vector<grafika*> grafiki_wysladowane_czastki_1;
	std::vector<grafika*> grafiki_wysladowane_czastki_2;

	std::vector<grafika*> grafiki_razem_czastki;

	int ogladany_czas = 0;
	float skala_obrazu = 1.0f;
	float wzmocnienie = 1.0f;

	std::vector<prob_t> prawdopodop;
	std::vector<fp_t> czasy;

	std::vector<statyczny_wektor<prob_t>> prawdopodob_wysladowane_1;
	std::vector<statyczny_wektor<prob_t>> prawdopodob_wysladowane_2;

	static constexpr uint32_t skalar_instancji = 1;
	static constexpr uint32_t liczba_wierzcholkow = 100 * skalar_instancji + 1;

	static constexpr uint32_t pozycja_poczatkowa_1 = 5 * skalar_instancji;
	static constexpr uint32_t pozycja_poczatkowa_2 = 95 * skalar_instancji;

	static constexpr uint32_t liczba_iteracji = 2000 * skalar_instancji;
	//static constexpr uint32_t liczba_iteracji = 1000;
	static constexpr uint32_t jak_czesto_zapisac = 1 * skalar_instancji; //* skalar_instancji;
	//static constexpr uint32_t jak_czesto_zapisac = liczba_iteracji - 1; //* skalar_instancji;

	TMDQ transformata_pole = H;
	TMDQ transformata_oddzialywanie = HxH;//Fourier_4;

	graf przestrzen_1_czastka;
	spacer_losowy<zesp, TMDQ> spacer_1_czastka1;
	spacer_losowy<zesp, TMDQ> spacer_1_czastka2;

	graf przestrzen_2_czastki;
	spacer_losowy<zesp, TMDQ> spacer_2_czastki;

	__host__ linia_TMDQ()
		: nazwa_okna("Spojrzenie 2 czastki linia kwantowe")
		, przestrzen_1_czastka(graf_lini(liczba_wierzcholkow))
		, spacer_1_czastka1(spacer_linia<zesp, TMDQ>(liczba_wierzcholkow, transformata_pole, transformata_pole, &przestrzen_1_czastka))
		, spacer_1_czastka2(spacer_1_czastka1)
		, przestrzen_2_czastki(przestrzen_1_czastka.tensorowy(2))
		, spacer_2_czastki(spacer_linia_2_czastki<zesp, TMDQ>(liczba_wierzcholkow, transformata_oddzialywanie, transformata_pole, &przestrzen_2_czastki))
		{

		reset_spacery();
		policz_gpu();
		przygotuj_grafiki();
	}

	void policz_cpu() {
		printf("CPU start\n");
		for (uint32_t i = 0; i < liczba_iteracji; i++) {
			spacer_1_czastka1.iteracja_na_cpu();
			if (i % jak_czesto_zapisac == 0) {
				spacer_1_czastka1.zapisz_iteracje();
			}
			spacer_1_czastka1.dokoncz_iteracje(dt);
		}

		for (uint32_t i = 0; i < liczba_iteracji; i++) {
			spacer_1_czastka2.iteracja_na_cpu();
			if (i % jak_czesto_zapisac == 0) {
				spacer_1_czastka2.zapisz_iteracje();
			}
			spacer_1_czastka2.dokoncz_iteracje(dt);
		}

		for (uint32_t i = 0; i < liczba_iteracji; i++) {
			spacer_2_czastki.iteracja_na_cpu();
			if (i % jak_czesto_zapisac == 0) {
				spacer_2_czastki.zapisz_iteracje();
			}
			spacer_2_czastki.dokoncz_iteracje(dt);
		}

		printf("CPU koniec\n");
	}

	void policz_gpu() {
		printf("CUDA start\n");
		spacer_1_czastka1.zbuduj_na_cuda();
		proste_iteracje_na_gpu<zesp, TMDQ>(spacer_1_czastka1, 1.0, liczba_iteracji, 70, 300, jak_czesto_zapisac);
		spacer_1_czastka1.zburz_na_cuda();

		spacer_1_czastka2.zbuduj_na_cuda();
		proste_iteracje_na_gpu<zesp, TMDQ>(spacer_1_czastka2, 1.0, liczba_iteracji, 70, 300, jak_czesto_zapisac);
		spacer_1_czastka2.zburz_na_cuda();

		CZAS_INIT
		CZAS_START
		spacer_2_czastki.zbuduj_na_cuda();
		proste_iteracje_na_gpu<zesp, TMDQ>(spacer_2_czastki, 1.0, liczba_iteracji, 70, 300, jak_czesto_zapisac);
		spacer_2_czastki.zburz_na_cuda();
		CZAS_STOP
		printf("CUDA koniec\n");
	}

	spacer_losowy<zesp, TMDQ>& spacer(){
		return spacer_2_czastki;
	}

	__host__ uint64_t liczba_zapamietanych_iteracji() {
		return spacer().iteracje_zapamietane.rozmiar;
	}

	__host__ void przygotuj_grafiki(uint64_t rozmiar_przed = std::numeric_limits<uint64_t>::max()) {
		ASSERT_Z_ERROR_MSG(liczba_wierzcholkow * liczba_wierzcholkow ==
		spacer().trwale.liczba_wierzcholkow(), "Tego spaceru nie da sie przedstawic jako kwadrat\n");

		if (rozmiar_przed == std::numeric_limits<uint64_t>::max()) rozmiar_przed = grafiki_2_czastki.size();
		grafiki_2_czastki.resize(liczba_zapamietanych_iteracji());
		grafiki_2_czastki_1.resize(liczba_zapamietanych_iteracji());
		grafiki_2_czastki_2.resize(liczba_zapamietanych_iteracji());

		grafiki_wysladowane_czastki_1.resize(liczba_zapamietanych_iteracji());
		grafiki_wysladowane_czastki_2.resize(liczba_zapamietanych_iteracji());
		grafiki_razem_czastki.resize(liczba_zapamietanych_iteracji());

		prawdopodop.resize(liczba_zapamietanych_iteracji());
		czasy.resize(liczba_zapamietanych_iteracji());

		prawdopodob_wysladowane_1.resize(liczba_zapamietanych_iteracji(), spacer_1_czastka1.trwale.liczba_kubelkow());
		prawdopodob_wysladowane_2.resize(liczba_zapamietanych_iteracji(), spacer_1_czastka2.trwale.liczba_kubelkow());

		for (uint64_t i = rozmiar_przed; i < liczba_zapamietanych_iteracji(); i++) {
			spacer::dane_iteracji<zesp>* iteracja = spacer_2_czastki.iteracje_zapamietane[i];
			grafiki_2_czastki[i] = grafika_P_kierunkow_dla_kraty_2D(spacer(), *iteracja,
				liczba_wierzcholkow, liczba_wierzcholkow, &(prawdopodop[i]), wzmocnienie);

			statyczny_wektor<prob_t>& wypelniany1 = prawdopodob_wysladowane_1[i];
			statyczny_wektor<prob_t>& wypelniany2 = prawdopodob_wysladowane_2[i];
			for(uint32_t j = 0; j < wypelniany1.rozmiar; j++){
				wypelniany1[j] = 0.0;
				wypelniany2[j] = 0.0;
			}

			for (uint32_t y = 0; y < liczba_wierzcholkow; y++) {
			for (uint32_t x = 0; x < liczba_wierzcholkow; x++) {
				spacer::wierzcholek& W = spacer_2_czastki.trwale
				.wierzcholki[y * liczba_wierzcholkow + x];

				for(uint8_t k = 0; k < W.liczba_kierunkow; k++){
					prob_t PP = P(iteracja->wartosci[W.start_wartosci + k]);

					wypelniany1[spacer_1_czastka1.trwale.wierzcholki[y].start_wartosci + ((k >> 1) & 1)] += PP;
					wypelniany2[spacer_1_czastka2.trwale.wierzcholki[x].start_wartosci + (k & 1)] += PP;
				}
			}}
			grafiki_wysladowane_czastki_1[i] = grafika_P_kierunkow_dla_kraty_2D(spacer_1_czastka1,
				wypelniany1, liczba_wierzcholkow, 1, nullptr, wzmocnienie);

			grafiki_wysladowane_czastki_2[i] = grafika_P_kierunkow_dla_kraty_2D(spacer_1_czastka2,
				wypelniany2, liczba_wierzcholkow, 1, nullptr, wzmocnienie);

			grafiki_razem_czastki[i] = grafika_P_pozycji_2_krata_2D(spacer_1_czastka1, wypelniany1, wypelniany2, liczba_wierzcholkow, 1, wzmocnienie);

			iteracja = spacer_1_czastka1.iteracje_zapamietane[i];
			grafiki_2_czastki_1[i] = grafika_P_kierunkow_dla_kraty_2D(spacer_1_czastka1,
				*iteracja, liczba_wierzcholkow, 1, nullptr, wzmocnienie);

			iteracja = spacer_1_czastka2.iteracje_zapamietane[i];
			grafiki_2_czastki_2[i] = grafika_P_kierunkow_dla_kraty_2D(spacer_1_czastka2,
				*iteracja, liczba_wierzcholkow, 1, nullptr, wzmocnienie);

			czasy[i] = iteracja->czas;
		}
	}

	__host__ void pokaz_wykresy() {
		if (ImPlot::BeginPlot("##Dane w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
			ImPlot::PlotInfLines("Vertical pomocnik", &czasy[ogladany_czas], 1);
			ImPlot::PlotLine("Prawdopodobienstwa suma", czasy.data(), prawdopodop.data(), (int)liczba_zapamietanych_iteracji());
			ImPlot::EndPlot();
		}
	}

	__host__ void pokaz_spacery(){
		ImGui::Begin(("Wykresy " + nazwa_okna).c_str());
		ImGui::Text("t = %lf", czasy[ogladany_czas]);
		plot_spacer_dla_kraty_2D(spacer(), ogladany_czas, przestrzen_2_czastki,
			grafiki_2_czastki[ogladany_czas], liczba_wierzcholkow, liczba_wierzcholkow,
			skala_obrazu, "Spacer dwuczasteczkowy widok globalny");

		ImGui::SameLine();

		plot_spacer_dla_kraty_2D(spacer_1_czastka1, ogladany_czas, przestrzen_1_czastka,
			grafiki_2_czastki_1[ogladany_czas], liczba_wierzcholkow, 1,
			skala_obrazu, "Spacer pojedynczej czastki 1");

		ImGui::SameLine();

		plot_spacer_dla_kraty_2D(spacer_1_czastka2, ogladany_czas, przestrzen_1_czastka,
			grafiki_2_czastki_2[ogladany_czas], liczba_wierzcholkow, 1,
			skala_obrazu, "Spacer pojedynczej czastki 2");


		ImVec2 bmin(0.0, 0.0);
		ImVec2 bmax((float)liczba_wierzcholkow, (float)1);
		ImVec2 uv0(0.0, 0.0);
		ImVec2 uv1(1.0, -1.0); // bo tak tworze osie przy tworzeniu grafu

		if (ImPlot::BeginPlot("Dwie czastki po wysladowaniu", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
			ImPlot::SetupAxes("pozycja", "1");
			ImPlot::PlotImage("##P", (ImTextureID)(intptr_t)
				(grafiki_razem_czastki[ogladany_czas]->texture)
				, bmin, bmax, uv0, uv1);
			ImPlot::EndPlot();
		}
		ImGui::SameLine();


		if (ImPlot::BeginPlot("Wysladowana czastka 1", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
			ImPlot::SetupAxes("pozycja", "1");
			ImPlot::PlotImage("##P", (ImTextureID)(intptr_t)
				(grafiki_wysladowane_czastki_1[ogladany_czas]->texture)
				, bmin, bmax, uv0, uv1);
			ImPlot::EndPlot();
		}
		ImGui::SameLine();

		if (ImPlot::BeginPlot("Wysladowana czastka 2", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
			ImPlot::SetupAxes("pozycja", "1");
			ImPlot::PlotImage("##P", (ImTextureID)(intptr_t)
				(grafiki_wysladowane_czastki_2[ogladany_czas]->texture)
				, bmin, bmax, uv0, uv1);
			ImPlot::EndPlot();
		}
		ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ContextMenuInBody;

		if (ImGui::BeginTable("##cos", 2, flags)) {
			ImGui::TableNextRow();

			ImGui::TableSetColumnIndex(0);
			ImGui::Text("Transformata pole");

			ImGui::TableSetColumnIndex(1);
			ImGui::Text("Transformata oddzialywanie");



			ImGui::TableNextRow();

			ImGui::TableSetColumnIndex(0);
			pokaz_transformate(transformata_pole);

			ImGui::TableSetColumnIndex(1);
			pokaz_transformate(transformata_oddzialywanie);

			ImGui::EndTable();
		}

		ImGui::End();
	}

	__host__ void reset_spacery(){
		spacer_1_czastka1.reset();
		spacer_1_czastka2.reset();
		spacer_2_czastki.reset();

		spacer_1_czastka1.iteracjaA[spacer_1_czastka1.trwale.wierzcholki[liczba_wierzcholkow / 2].start_wartosci] = zero(zesp());
		spacer_1_czastka2.iteracjaA[spacer_1_czastka2.trwale.wierzcholki[liczba_wierzcholkow / 2].start_wartosci] = zero(zesp());

		spacer_1_czastka1.iteracjaA[spacer_1_czastka1.trwale.wierzcholki[pozycja_poczatkowa_1].start_wartosci] = (zesp(1.0, 1.0) / std::sqrt(fp_t(2.0)));
		spacer_1_czastka2.iteracjaA[spacer_1_czastka2.trwale.wierzcholki[pozycja_poczatkowa_2].start_wartosci + 1] = (zesp(1.0, 1.0) / std::sqrt(fp_t(2.0)));

		spacer_2_czastki.iteracjaA[0] = zero(zesp());
		spacer_2_czastki.iteracjaA[spacer_2_czastki.trwale.wierzcholki[
			pozycja_poczatkowa_1 * liczba_wierzcholkow + pozycja_poczatkowa_2]
			.start_wartosci + 1] = zesp(1.0, 1.0) * zesp(1.0, 1.0) / 2.0;
	}

	__host__ void reset_grafiki(){
		for (auto g : grafiki_2_czastki) delete g;
		for (auto g : grafiki_2_czastki_1) delete g;
		for (auto g : grafiki_2_czastki_2) delete g;
		for (auto g : grafiki_wysladowane_czastki_1) delete g;
		for (auto g : grafiki_wysladowane_czastki_2) delete g;
		for (auto g : grafiki_razem_czastki) delete g;
		grafiki_2_czastki.resize(0);
		grafiki_2_czastki_1.resize(0);
		grafiki_2_czastki_2.resize(0);
		grafiki_wysladowane_czastki_1.resize(0);
		grafiki_wysladowane_czastki_2.resize(0);
		grafiki_razem_czastki.resize(0);
	}

	__host__ void kolejne_transformaty(){
		reset_spacery();
		transformata_pole = losowa_transformata(2);
		transformata_oddzialywanie = losowa_transformata(4);

		//TMDQ transformata_pole = T_rozne_pozycje;
		//TMDQ transformata_oddzialywanie = T_te_same_pozycje;

		TMDQ transformata_pole_ztensorowana = tensor(transformata_pole, transformata_pole);
		TMDQ transformata_oddzialywanie_calkowite = mnoz(transformata_oddzialywanie, transformata_pole_ztensorowana);

		spacer_1_czastka1.trwale.zamien_transformate(0, transformata_pole);
		spacer_1_czastka1.trwale.zamien_transformate(1, transformata_pole); // boki

		spacer_1_czastka2.trwale.zamien_transformate(0, transformata_pole);
		spacer_1_czastka2.trwale.zamien_transformate(1, transformata_pole); // boki

		spacer_2_czastki.trwale.zamien_transformate(0, transformata_pole_ztensorowana);
		spacer_2_czastki.trwale.zamien_transformate(1, transformata_oddzialywanie_calkowite);
		policz_gpu();
		reset_grafiki();
		przygotuj_grafiki();

	}

	__host__ ~linia_TMDQ() {
		for (auto g : grafiki_2_czastki) delete g;
		for (auto g : grafiki_2_czastki_1) delete g;
		for (auto g : grafiki_2_czastki_2) delete g;
		for (auto g : grafiki_wysladowane_czastki_1) delete g;
		for (auto g : grafiki_wysladowane_czastki_2) delete g;
		for (auto g : grafiki_razem_czastki) delete g;
	}
};

struct przegladacz_instancji : linia_TMDQ {
	static constexpr bool zapisz = false;
	static constexpr bool daj_nowa_instancje = true;
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

	void tick()	{
		if (okres_pokazu_slajdow < 0.95f) {
			double czas = glfwGetTime();
			if (czas > (ostatni_czas_odswiezenia + (double)okres_pokazu_slajdow)) {
				if(nastepny_obraz() && daj_nowa_instancje){
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

	void pokaz_kontrolki(){
		ImGui::Begin(("Kontrolki: "+nazwa_okna).c_str());
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
		if(ImGui::Button("Zapisz screenshot")){
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
		if(ogladany_czas >= liczba_zapamietanych_iteracji()){
			ogladany_czas = 0;
			return true;	
		}
		return false;
	}

};



}
 