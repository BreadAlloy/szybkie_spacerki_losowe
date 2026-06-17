#pragma once

#include "spacer_losowy.h"
#include "transformaty_wyspecializowane.h"
#include "grafika.h"
#include "zapamietywacz.h"

#include "transformata_fouriera.cuh"
#include "rozniczka.cuh"

struct czy_czasteczka_okno{
	const uint32_t liczba_wierzcholkow_boku;
	const std::string nazwa_okna;

	czy_jest_czastka dane;
	
	std::vector<grafika*> grafiki_iteracji;
	std::vector<std::vector<grafika*>> grafiki_iteracji_kierunki;

	std::vector<double> normy_skalarow;
	std::vector<double> estymowana_masa;
	std::vector<double> prawdopodop;
	std::vector<double> czasy;

	spacer_losowy<zesp, TMCQ>* spacer;
	graf* przestrzen_ptr;

	int pokazywana_grafika = 0;
	float skala_obrazu = 1.0f;
	float wzmocnienie = 1.0f;
	float okres_pokazu_slajdow = 1.0f;
	double ostatni_czas_odswiezenia = glfwGetTime();

	czy_czasteczka_okno(spacer_losowy<zesp, TMCQ>* spacer, std::string nazwa_okna, uint32_t liczba_wierzcholkow_boku, graf* przestrzen_ptr)
	: dane(*spacer), nazwa_okna(nazwa_okna), spacer(spacer), liczba_wierzcholkow_boku(liczba_wierzcholkow_boku), przestrzen_ptr(przestrzen_ptr){
		przygotuj_grafiki();
	}

	void przygotuj_grafiki(){
		for (auto& g : grafiki_iteracji) {
			delete g;
		}

		estymowana_masa.resize(dane.dla_iteracji.size());
		normy_skalarow.resize(dane.dla_iteracji.size());
		grafiki_iteracji.resize(dane.dla_iteracji.size());
		czasy.resize(dane.dla_iteracji.size());
		prawdopodop.resize(dane.dla_iteracji.size());

		for (uint64_t i = 0; i < dane.dla_iteracji.size(); i++) {
			statyczny_wektor<zesp>& blad = dane.dla_iteracji[i].blad;
			spacer::dane_iteracji<zesp>& iteracja = *(spacer->iteracje_zapamietane[i]);
			czasy[i] = iteracja.czas;
			grafiki_iteracji[i] = grafika_P_kierunkow_dla_kraty_2D(*spacer,
				blad, liczba_wierzcholkow_boku, liczba_wierzcholkow_boku, &(prawdopodop[i]), 1.0f);
			normy_skalarow[i] = dane.dla_iteracji[i].x.norm();
			estymowana_masa[i] = 1.0 / normy_skalarow[i];
		}
	}

	bool pokaz_okno(){
		bool ret = true;
		ImGui::Begin(nazwa_okna.c_str(), &ret);
		ImGui::SliderFloat("Rozmiar grafiki", &skala_obrazu, 0.0f, 10.0f);
		ImGui::SliderFloat("Wzmocnienie", &wzmocnienie, 1.0f, 10.0f);
		ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, -0.01f, 1.0f);

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

		if (grafiki_iteracji.size() != 0) {
			ImGui::SliderInt("Pokazywana grafika", &pokazywana_grafika, 0, grafiki_iteracji.size() - 1UL);
			grafika* G = grafiki_iteracji[pokazywana_grafika];
			plot_spacer_dla_kraty_2D(*spacer, dane.dla_iteracji[pokazywana_grafika].blad, *przestrzen_ptr, G, liczba_wierzcholkow_boku, liczba_wierzcholkow_boku, skala_obrazu, "Blad rozwiazania rownania rozniczkowego");
			ImGui::SameLine();
			if (ImPlot::BeginPlot("##Blad", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
				ImPlot::PlotInfLines("Biezacy czas", &czasy[pokazywana_grafika], 1);
				ImPlot::PlotLine("Blad calkowity", czasy.data(), prawdopodop.data(), (int)czasy.size());
				ImPlot::PlotLine("Norma skalaru", czasy.data(), normy_skalarow.data(), (int)czasy.size());
				ImPlot::PlotLine("Estymowana masa", czasy.data(), estymowana_masa.data(), (int)czasy.size());
				ImPlot::PlotLineLepsze("Skalar(Re)", czasy.data(), &(dane.dla_iteracji.data()->x.Re), (int)czasy.size(), 0, 0, sizeof(podobienstwo_liniowe));
				ImPlot::PlotLineLepsze("Skalar(Im)", czasy.data(), &(dane.dla_iteracji.data()->x.Im), (int)czasy.size(), 0, 0, sizeof(podobienstwo_liniowe));
				ImPlot::EndPlot();
			}
		}
		ImGui::End();
		return ret;
	}
};

template <typename transformata> // nie obs³uguje klasycznych
struct okno_przegladania{
	const uint32_t liczba_wierzcholkow_boku;
	const std::string nazwa_okna;

	int co_ile_zapisz = 100;
	int liczba_iteracji = 0;
	int pokazywana_grafika = 0;
	float skala_obrazu = 1.0f;
	float wzmocnienie = 1.0f;
	float okres_pokazu_slajdow = 1.0f;
	double ostatni_czas_odswiezenia = glfwGetTime();
	bool pedowo = false;

	graf* przestrzen_ptr = nullptr;
	spacer_losowy<zesp, transformata> poczatkowy;
	spacer_losowy<zesp, transformata> spacer;
	czy_czasteczka_okno* czy_czastka = nullptr;

	std::vector<grafika*> grafiki_iteracji;
	std::vector<std::vector<grafika*>> grafiki_iteracji_kierunki;

	std::vector<double> prawdopodop;
	std::vector<double> czasy;

	okno_przegladania(std::string nazwa_okna, float rozmiar_grafiki, uint32_t liczba_wierzcholkow_boku, graf* przestrzen_ptr, spacer_losowy<zesp, transformata>& spacer)
	: nazwa_okna(nazwa_okna)
	, skala_obrazu(rozmiar_grafiki)
	, liczba_wierzcholkow_boku(liczba_wierzcholkow_boku)
	, przestrzen_ptr(przestrzen_ptr)
	, poczatkowy(spacer)
	, spacer(poczatkowy)
	{
		przelicz();
	}

	__host__ void przygotuj_grafiki() {
		for (auto& g : grafiki_iteracji) {
			delete g;
		}

		grafiki_iteracji.resize(spacer.iteracje_zapamietane.rozmiar);
		grafiki_iteracji_kierunki.resize(spacer.iteracje_zapamietane.rozmiar);
		czasy.resize(spacer.iteracje_zapamietane.rozmiar);
		prawdopodop.resize(spacer.iteracje_zapamietane.rozmiar);

		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			spacer::dane_iteracji<zesp>& iteracja = *(spacer.iteracje_zapamietane[i]);
			czasy[i] = iteracja.czas;
			grafiki_iteracji[i] = grafika_P_kierunkow_dla_kraty_2D(spacer,
				iteracja, liczba_wierzcholkow_boku, liczba_wierzcholkow_boku, &(prawdopodop[i]), wzmocnienie);
		}
	}

	void przelicz(){
		printf("\nLiczba iteracji:%d\n", liczba_iteracji);
		spacer = poczatkowy;

		CZAS_INIT
		spacer.zbuduj_na_cuda();

		printf("GPU start\n");
		CZAS_START
		iteracje_na_gpu<zesp, TMCQ>(spacer, dt, liczba_iteracji, 30, 500, co_ile_zapisz, 1, liczba_iteracji + 1);
		CZAS_STOP
		printf("GPU koniec\n");

		spacer.zburz_na_cuda();

		przygotuj_grafiki();
	}

	void policz_czy_czastka(){
		if(czy_czastka != nullptr){
			delete czy_czastka;
		}
		czy_czastka = new czy_czasteczka_okno(&spacer, nazwa_okna + ": czy czastka", liczba_wierzcholkow_boku, przestrzen_ptr);
	}

	void fourieruj(bool odwrocona){
		fft_3d_po_jednym fourierowacz(liczba_wierzcholkow_boku, liczba_wierzcholkow_boku, 4);
		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			if(odwrocona){
				fourierowacz.fft_inv(spacer.iteracje_zapamietane[i]->wartosci);
			} else {
				fourierowacz.fft(spacer.iteracje_zapamietane[i]->wartosci);
			}
		}
	}

	void rozniczkuj_po_przstrzeni(){
		spacer.zbuduj_na_cuda();
		rozniczka_po_przestrzeni rozniczkowacz(&(spacer.trwale));
		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			statyczny_wektor<zesp>& dane = spacer.iteracje_zapamietane[i]->wartosci;
			dane.cuda_malloc();
			dane.cuda_zanies(rozniczkowacz.stream);
			rozniczkowacz.rozniczkuj(&(spacer.lokalizacja_na_device->trwale), spacer.iteracje_zapamietane[i]->wartosci);
			dane.cuda_przynies(rozniczkowacz.stream);
			dane.cuda_free();
		}
		spacer.zburz_na_cuda();
	}

	void rozniczkuj_po_czasie() {
		spacer.zbuduj_na_cuda();
		rozniczka_po_czasie rozniczkowacz(&(spacer.trwale));
		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			statyczny_wektor<zesp>& dane = spacer.iteracje_zapamietane[i]->wartosci;
			dane.cuda_malloc();
			dane.cuda_zanies(rozniczkowacz.stream);
			rozniczkowacz.rozniczkuj(&(spacer.lokalizacja_na_device->trwale), spacer.iteracje_zapamietane[i]->wartosci);
			dane.cuda_przynies(rozniczkowacz.stream);
			dane.cuda_free();
		}
		spacer.zburz_na_cuda();
	}

	void laplasuj_po_przstrzeni() {
		spacer.zbuduj_na_cuda();
		laplasjan_po_przestrzeni laplasowacz(&(spacer.trwale));
		for (uint64_t i = 0; i < spacer.iteracje_zapamietane.rozmiar; i++) {
			statyczny_wektor<zesp>& dane = spacer.iteracje_zapamietane[i]->wartosci;
			dane.cuda_malloc();
			dane.cuda_zanies(laplasowacz.stream);
			laplasowacz.laplasuj(&(spacer.lokalizacja_na_device->trwale), spacer.iteracje_zapamietane[i]->wartosci);
			dane.cuda_przynies(laplasowacz.stream);
			dane.cuda_free();
		}
		spacer.zburz_na_cuda();
	}

	bool pokaz_okno(){
		bool ret = true;
		ImGui::Begin(nazwa_okna.c_str(), &ret);
		ImGui::SliderFloat("Rozmiar grafiki", &skala_obrazu, 0.0f, 10.0f);
		ImGui::SliderInt("Liczba iteracji", &liczba_iteracji, 1, 100000);
		ImGui::SliderInt("Co ile zapisac", &co_ile_zapisz, 1, liczba_iteracji);
		ImGui::SliderFloat("Wzmocnienie", &wzmocnienie, 1.0f, 10.0f);
		ImGui::SliderFloat("Okres pokazu slajdow(1.0 to brak pokazu)", &okres_pokazu_slajdow, -0.01f, 1.0f);

		if(ImGui::Button("Przelicz")){
			przelicz();
		}
		ImGui::SameLine();
		if (ImGui::Button("Wygeneruj grafiki")) {
			przygotuj_grafiki();
		}
		ImGui::SameLine();
		if(ImGui::Checkbox("Pedowo", &pedowo)){
			fourieruj(!pedowo);
			przygotuj_grafiki();
		}
		ImGui::SameLine();
		if (ImGui::Button("Rozniczkuj po przestrzeni")) {
			rozniczkuj_po_przstrzeni();
			przygotuj_grafiki();
		}
		ImGui::SameLine();
		if (ImGui::Button("Laplasuj po przestrzeni")) {
			laplasuj_po_przstrzeni();
			przygotuj_grafiki();
		}
		ImGui::SameLine();
		if (ImGui::Button("Rozniczkuj po czasie")) {
			rozniczkuj_po_czasie();
			przygotuj_grafiki();
		}
		ImGui::SameLine();
		if (ImGui::Button("Czy jest czasteczka")) {
			policz_czy_czastka();
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

		if(grafiki_iteracji.size() != 0){
			ImGui::SliderInt("Pokazywana grafika", &pokazywana_grafika, 0, grafiki_iteracji.size() - 1UL);
			ImGui::Text("Spacer: Prawdopodobienstwo poprzedniej:%lf, Zaabsorbowane poprzedniej: %lf, Norma poprzedniej: %lf",
				spacer.iteracje_zapamietane[pokazywana_grafika]->prawdopodobienstwo_poprzedniej,
				spacer.iteracje_zapamietane[pokazywana_grafika]->zaabsorbowane_poprzedniej,
				spacer.iteracje_zapamietane[pokazywana_grafika]->norma_poprzedniej_iteracji
			);
			grafika* G = grafiki_iteracji[pokazywana_grafika];
			plot_spacer_dla_kraty_2D(spacer, pokazywana_grafika, *przestrzen_ptr, G, liczba_wierzcholkow_boku, liczba_wierzcholkow_boku, skala_obrazu, "spacer");
			ImGui::SameLine();
			if (ImPlot::BeginPlot("##Dane w spacerze", ImVec2(skala_obrazu * 200.0f, skala_obrazu * 200.0f))) {
				ImPlot::PlotInfLines("Vertical pomocnik cpu", &czasy[pokazywana_grafika], 1);

				ImPlot::PlotLine("Pozostale cpu", czasy.data(), prawdopodop.data(), (int)czasy.size());
				ImPlot::EndPlot();
			}
		}
		ImGui::End();
		if(czy_czastka != nullptr){
			if(!czy_czastka->pokaz_okno()){
				delete czy_czastka;
				czy_czastka = nullptr;
			}
		}
		return ret;
	}

	~okno_przegladania(){
		if(czy_czastka != nullptr){
			delete czy_czastka;
			czy_czastka = nullptr;
		}
	}

};

template <typename transformata> // nie obs³uguje klasycznych
struct preview_zapisanych{
	const uint32_t indeks_grafiki_do_preview = 2;
	const float rozmiar_guzika = 125.0f + 50.0f; // 50 to padding z rêki estymowany

	const std::string nazwa_okna;

	std::vector<grafika*> grafiki;

	preview_zapisanych(std::string folder, uint32_t ile_zapisanych)
	: nazwa_okna("Preview iteracji w folderze: " + folder)
	{
		grafiki.resize(ile_zapisanych);
		for (uint64_t i = 0; i < ile_zapisanych; i++) {
			grafika* nowa = new grafika((folder + "//transformata_" + std::to_string(i) +
				"-grafika_" + std::to_string(indeks_grafiki_do_preview) + ".bmp").c_str());
			grafiki[i] = nowa;
		}
	}

	int pokaz_swoje(){
		int ret = -1;
		if(ImGui::Begin(nazwa_okna.c_str())){
			float szerokosc_okna = ImGui::GetWindowSize().x;
			uint64_t ile_grafik_na_wiersz = MAX((uint64_t)(szerokosc_okna / rozmiar_guzika - 0.5f), 1UL);

			ImVec2 uv0(0.0f, 0.0f);
			ImVec2 uv1(1.0f, -1.0f);
			ImVec2 size(rozmiar_guzika, rozmiar_guzika);
			for(uint64_t i = 0; i < grafiki.size(); i++){
				grafika* G = grafiki[i];
				if(ImGui::ImageButton(("Preview" + std::to_string(i)).c_str(), (ImTextureID)(intptr_t)(G->texture), size, uv0, uv1)){
					ret = (int)i;
				}
				if((i + 1UL) % ile_grafik_na_wiersz != 0) ImGui::SameLine();
			}
		}
		ImGui::End();
		return ret;
	}

	~preview_zapisanych(){
		for (auto G : grafiki) {
			delete G;
		}
	}
};

template <typename transformata> // nie obs³uguje klasycznych
struct przejrzenie_reczne{
	// pola ustawiane przy kompilacji
	const uint32_t liczba_wierzcholkow_boku = 501;
	const uint32_t liczba_podgrafik = 5;

	const std::string nazwa_okna;
	const std::string folder;

	std::vector<okno_przegladania<transformata>*> podokna;
	zapamietywacz::baza_transformat<TMCQ> transformaty_przejrzane;
	std::vector<grafika*> grafiki_zapamietane;
	preview_zapisanych<transformata> preview;

	int wybrana = 0;
	float rozmiar_wykresu = 1.0f;
	bool wszystkie_na_raz = true;
	bool przejdz_do_nastepnej = false;
	bool pokaz_slajdow = true;
	bool pokaz_preview = true;
	int pokazywana_grafika = 0;	
	float okres_pokazu_slajdow = 2.0f;
	double ostatni_czas_odswiezenia = 0.0;

	transformata_macierz<double> przecietna_transformata;

	graf przestrzen;
	spacer_losowy<zesp, TMCQ> spacer;

	przejrzenie_reczne(std::string folder)
	: folder(folder)
	, nazwa_okna("Dokladniejsze przejrzenie: " + folder)
	, transformaty_przejrzane(folder)
	, preview(folder, transformaty_przejrzane.transformaty.size())
	, przecietna_transformata((uint8_t)4)
	, przestrzen(graf_krata_2D_cykl(liczba_wierzcholkow_boku))
	, spacer(spacer_krata_2D_cykl<zesp, TMCQ>(liczba_wierzcholkow_boku, tensor(X, H), &przestrzen))
	{

		spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = zero(zesp());
		spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci] = jeden(zesp()) / sqrt(2.0);
		spacer.iteracjaA[spacer.trwale.wierzcholki[(liczba_wierzcholkow_boku * liczba_wierzcholkow_boku) / 2].start_wartosci + 2] = jeden(zesp()) / sqrt(2.0);

		przygotuj_do_szybkiego_podgladu();
	}

	void przygotuj_do_szybkiego_podgladu(){
		pokazywana_grafika = 0;

		for(auto G : grafiki_zapamietane){
			delete G;
		}

		for(uint8_t i = 0; i < (uint8_t)4; i++){
			for (uint8_t j = 0; j < (uint8_t)4; j++) {
				przecietna_transformata(i, j) = P(transformaty_przejrzane.transformaty[wybrana](i, j));
			}
		}

		grafiki_zapamietane.resize(liczba_podgrafik);

		for(uint32_t i = 0; i < liczba_podgrafik; i++){
			grafika* nowa = new grafika((folder + "//transformata_" + std::to_string(wybrana) +
			"-grafika_" + std::to_string(i) + ".bmp").c_str());
			grafiki_zapamietane[i] = nowa;
		}
	}

	void nowe_okno(){
		spacer.trwale.zamien_transformate(0, transformaty_przejrzane.transformaty[wybrana]);
		spacer.trwale.zamien_transformate(1, transformaty_przejrzane.transformaty[wybrana]);

		okno_przegladania<transformata>* nowe_okno = new okno_przegladania<transformata>(
		nazwa_okna + " transformata: " + std::to_string(wybrana), rozmiar_wykresu * 1.5f,
		liczba_wierzcholkow_boku, &przestrzen, spacer);
		podokna.push_back(nowe_okno);
	}
	
	void pokaz_podokna(){
		for(uint64_t i = 0; i < podokna.size(); i++){
			if(!podokna[i]->pokaz_okno()){
				delete podokna[i];
				podokna[i] = podokna[podokna.size() - 1UL];
				podokna.resize(podokna.size() - 1UL);
			}
		}
	}

	void pokaz_okno(){
		ImGui::Begin(nazwa_okna.c_str());

		int poprzednia_wybrana = wybrana;
		ImGui::InputInt("Instancja zapamietana", &wybrana, 1, 10);
		ImGui::SliderInt("Slider do wybierania", &wybrana, 0, transformaty_przejrzane.transformaty.size() - 1);
		wybrana = MIN((uint64_t)wybrana, transformaty_przejrzane.transformaty.size() - 1UL);
		ImGui::SliderFloat("Rozmiar wykresu", &rozmiar_wykresu, 1.0f, 8.0f);
		ImGui::Checkbox("Pokaz preview", &pokaz_preview);
		ImGui::SameLine();
		ImGui::Checkbox("Wszystkie na raz", &wszystkie_na_raz);
		if(!wszystkie_na_raz){
			ImGui::SameLine();
			ImGui::Checkbox("Przejdz do nastepnej", &przejdz_do_nastepnej);
			ImGui::SameLine();
			ImGui::Checkbox("Pokaz slajdow", &pokaz_slajdow);
			if (pokaz_slajdow) {
				ImGui::SliderFloat("Okres pokazu slajdow", &okres_pokazu_slajdow, -0.0001f, 10.0f);
			}
		}

		if (poprzednia_wybrana != wybrana) {
			przygotuj_do_szybkiego_podgladu();
		}

		pokaz_transformate<zesp>(transformaty_przejrzane.transformaty[wybrana]);
		pokaz_transformate<double>(przecietna_transformata);

		if(ImGui::Button("Przyjrzyj sie dokladniej")){
			nowe_okno();
		}
		if(wszystkie_na_raz) {
			if (ImPlot::BeginPlot("Podgrafiki", ImVec2(rozmiar_wykresu * 200.0f * float(liczba_podgrafik - 2), rozmiar_wykresu * 200.0f), ImPlotFlags_Equal)) {
				ImVec2 uv0(0.0, 0.0);
				ImVec2 uv1(1.0, -1.0);
				for (uint32_t i = 0; i < liczba_podgrafik; i++) {
					grafika* G = grafiki_zapamietane[i];
					float offset = float(i * G->width) + float(i)*20.0f;
					ImVec2 bmin(0.0 + offset, 0.0);
					ImVec2 bmax((float)G->width + offset, (float)G->height);
					ImPlot::PlotImage(("Podgrafika: " + std::to_string(i)).c_str(), (ImTextureID)(intptr_t)(G->texture), bmin, bmax, uv0, uv1);
				}
				ImPlot::EndPlot();
			}
		} else {
			ImGui::InputInt("Pokazywana podgrafika", &pokazywana_grafika, 1, 3);
			ImGui::SliderInt("Wybrana podgrafika", &pokazywana_grafika, 0, liczba_podgrafik - 1);

			if(pokaz_slajdow){
				double czas = glfwGetTime();
				if (czas > (ostatni_czas_odswiezenia + (double)okres_pokazu_slajdow)) {
					pokazywana_grafika++;
					ostatni_czas_odswiezenia = czas;
				}
			} else {
				ostatni_czas_odswiezenia = glfwGetTime();
			}

			poprzednia_wybrana = wybrana;
			if(przejdz_do_nastepnej){
				if(pokazywana_grafika >= (int32_t)liczba_podgrafik){
					wybrana++;
				}
				if(pokazywana_grafika < 0){
					wybrana--;
				}
				wybrana = MIN((uint64_t)wybrana, transformaty_przejrzane.transformaty.size() - 1UL);
				if (poprzednia_wybrana != wybrana) {
					przygotuj_do_szybkiego_podgladu();
				}
			}			

			pokazywana_grafika = pokazywana_grafika % liczba_podgrafik;

			if (ImPlot::BeginPlot("Podgrafika", ImVec2(rozmiar_wykresu * 200.0f * 1.5f, rozmiar_wykresu * 200.0f * 1.5f), ImPlotFlags_Equal)) {
				ImVec2 uv0(0.0f, 0.0f);
				ImVec2 uv1(1.0f, -1.0f);
				grafika* G = grafiki_zapamietane[pokazywana_grafika];
				ImVec2 bmin(0.0f, 0.0f);
				ImVec2 bmax((float)G->width, (float)G->height);
				ImPlot::PlotImage("grafika", (ImTextureID)(intptr_t)(G->texture), bmin, bmax, uv0, uv1);
				ImPlot::EndPlot();
			}
		}
		ImGui::End();
		if (pokaz_preview) {
			int temp = preview.pokaz_swoje();
			if ((temp != -1) && (temp != wybrana)) {
				wybrana = temp;
				przygotuj_do_szybkiego_podgladu();
			}
		}

		pokaz_podokna();
	}

	~przejrzenie_reczne(){
		for(auto o : podokna){
			delete o;
		}
		for (auto G : grafiki_zapamietane) {
			delete G;
		}
	}

};