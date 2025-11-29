#pragma once

#include <inttypes.h>
#include <vector>
#include <string>

#include "pomocne_funkcje.h"

typedef uint32_t ID_W; // typ indeksowania wierzcholków
typedef uint8_t ID_K;  // typ indeksowania kube³ków

namespace grafowe { //rzeczy potrzebne do definicji grafu

	struct krawedz { // skierowana

		ID_W index_wierzcholka_z;
		ID_K index_kubelka_z;
		/*

		z -----> do	

		*/
		ID_W index_wierzcholka_do;
		ID_K index_kubelka_do;

		krawedz(){}

		krawedz(const krawedz& kopiowana) {
			memcpy(this, &kopiowana, sizeof(krawedz));
		}

		krawedz& operator=(const krawedz& kopiowana){
			memcpy(this, &kopiowana, sizeof(krawedz));
			return *this;
		}

		krawedz(ID_W index_wierzcholka_z, ID_K index_kubelka_z,
			    ID_W index_wierzcholka_do, ID_K index_kubelka_do)
		: index_wierzcholka_z(index_wierzcholka_z), index_kubelka_z(index_kubelka_z)
		, index_wierzcholka_do(index_wierzcholka_do), index_kubelka_do(index_kubelka_do){}
	};

	//template<typename opis_type> // mo¿na dodaæ w przysz³oœci
	struct wierzcholek {
		std::vector<bool> czy_polaczone_z;
		/*

		z -----> do

		*/
		std::vector<bool> czy_polaczone_do;

		//opis_type		 opis; // mo¿na dodaæ w przysz³oœci
		std::string opis;

		wierzcholek(ID_K liczba_polaczen = 0, std::string opis = "") : 
		opis(opis) {
			daj_liczbe_polaczen(liczba_polaczen);
		}

		wierzcholek(const wierzcholek& kopiowany) : 
		czy_polaczone_z(kopiowany.czy_polaczone_z),
		czy_polaczone_do(kopiowany.czy_polaczone_do),
		opis(kopiowany.opis) {}
	
		wierzcholek& operator=(const wierzcholek& drugi){
			czy_polaczone_z = drugi.czy_polaczone_z;
			czy_polaczone_do = drugi.czy_polaczone_do;
			opis = drugi.opis;
			return *this;
		}

		ID_K liczba_polaczen() const{
			ASSERT_Z_ERROR_MSG(czy_polaczone_z.size() == czy_polaczone_do.size(), "Inna ilosc kubelkow z i do?!?!?!?");
			return (ID_K)czy_polaczone_z.size();
		}

		void daj_liczbe_polaczen(ID_K podawana_liczba_polaczen){
			ASSERT_Z_ERROR_MSG(liczba_polaczen() == 0, "wierzcholek z opisem:%s juz ma liczbe polaczen" SEP opis.c_str());
			czy_polaczone_z.resize(podawana_liczba_polaczen, false);
			czy_polaczone_do.resize(podawana_liczba_polaczen, false);
		}

		ID_K dodaj_liczbe_polaczen(ID_K dodawana_liczba_polaczen) {
			//zwraca ile jest po³¹czeñ po operacji
			ID_K stara_liczba_polaczen = liczba_polaczen(); 
			czy_polaczone_z.resize(stara_liczba_polaczen + dodawana_liczba_polaczen, false);
			czy_polaczone_do.resize(stara_liczba_polaczen + dodawana_liczba_polaczen, false);
			return liczba_polaczen();
		}

		// error bêdzie handlowaæ graf
		bool podepnij_z(ID_K index_kubelka_z){
			/* zwraca false jeœli jest ju¿ podpiête */
			
			bool biezacy_stan = czy_polaczone_z[index_kubelka_z];
			if(biezacy_stan) {		
				return false; // Jest ju¿ podpiête z tego. error bêdzie handlowaæ graf
			} else {
				czy_polaczone_z[index_kubelka_z] = true;
				return true;
			}
		}

		bool podepnij_do(ID_K index_kubelka_do) {
			/* zwraca false jeœli jest ju¿ podpiête */

			bool biezacy_stan = czy_polaczone_do[index_kubelka_do];
			if (biezacy_stan) {
				return false; // Jest ju¿ podpiête z tego. error bêdzie handlowaæ graf
			} else {
				czy_polaczone_do[index_kubelka_do] = true;
				return true;
			}
		}

		bool czy_wszystko_polaczone() const{
			bool ret = true;

			bool Z, DO;
			for(ID_K i = 0; i < liczba_polaczen(); i++){
				Z = czy_polaczone_z[i];
				DO = czy_polaczone_do[i];
				ret &= (Z & DO);
				if(!DO) printf("Wierzcholek z opisem %s, na kubelku %d nie jest polaczony do\n", opis.c_str(), i);
				if (!Z) printf("Wierzcholek z opisem %s, na kubelku %d nie jest polaczony z\n", opis.c_str(), i);

			}
			return ret;
		}

		ID_K gdzie_mozna_podpiac_z(){
			// znajduje najmniejszy index kube³ka mo¿liwy do podpiêcia z -1 gdy siê nie da
			for(ID_K i = 0; i < liczba_polaczen(); i++){
				if(!czy_polaczone_z[i]) return i;
			}
			return (ID_K)-1;
		}

		ID_K gdzie_mozna_podpiac_do() {
			// znajduje najmniejszy index kube³ka mo¿liwy do podpiêcia do -1 gdy siê nie da
			for (ID_K i = 0; i < liczba_polaczen(); i++) {
				if (!czy_polaczone_do[i]) return i;
			}
			return (ID_K)-1;
		}
	};
};

struct graf {
	// nie grzebaæ w modyfikowaæ tych pól
	std::vector<grafowe::wierzcholek> wierzcholki;
	std::vector<grafowe::krawedz> krawedzie; // skierowane

	graf(ID_W liczba_wierzcholkow = 0){
		// na pocz¹tku dla ka¿dego wierzcho³ka liczba po³¹czeñ wynosi 0.
		wierzcholki.resize(liczba_wierzcholkow);
	}

	graf(const graf& B) : krawedzie(B.liczba_krawedzi()), wierzcholki(B.liczba_wierzcholkow()){
		graf& A = *this; // nazwanie this A
		memcpy(A.krawedzie.data(), B.krawedzie.data(), sizeof(grafowe::krawedz) * A.liczba_krawedzi());
		for(ID_W i = 0; i < A.liczba_wierzcholkow(); i++){
			A.wierzcholki[i] = B.wierzcholki[i];
		}
	}

	ID_W liczba_wierzcholkow() const{
		return (ID_W)wierzcholki.size();
	}

	size_t liczba_krawedzi() const {
		return krawedzie.size();
	}

	ID_W dodaj_wierzcholek(ID_K liczba_polaczen_dodawanego){
		/* Zwraca index dodanego wierzcholka */
		ID_W ret = (ID_W)wierzcholki.size();
		wierzcholki.emplace_back(liczba_polaczen_dodawanego);
		return (ID_W)ret;
	}

	void zdefiniuj_liczbe_polaczen(ID_W index_wierzcholka, ID_K liczba_polaczen){
		wierzcholki[index_wierzcholka].daj_liczbe_polaczen(liczba_polaczen);
	}

	void update_opis(ID_W index_wierzcholka, std::string nowy_opis){
		wierzcholki[index_wierzcholka].opis = nowy_opis;
	}

	ID_K dodaj_polaczenia(ID_W index_wierzcholka, ID_K liczba_dodawanych_polaczen){
		return wierzcholki[index_wierzcholka].dodaj_liczbe_polaczen(liczba_dodawanych_polaczen);
	}

	void dodaj_krawedz(ID_W index_wierzcholka_z, ID_K  index_kubelka_z,
					  ID_W index_wierzcholka_do, ID_K  index_kubelka_do){
		// skierowana krawedz
		bool czy_mozna_z = wierzcholki[index_wierzcholka_z].podepnij_z(index_kubelka_z);
		bool czy_mozna_do = wierzcholki[index_wierzcholka_do].podepnij_do(index_kubelka_do);
		if(!czy_mozna_z || !czy_mozna_do){
			// nie jest dobrze
			ASSERT_Z_ERROR_MSG(false, "Krawedz z lokalizacji: %d|%d do: %d|%d nie jest mozliwa\n" SEP index_wierzcholka_z SEP index_kubelka_z SEP index_wierzcholka_do SEP index_kubelka_do);
		} else {
			// wszystko jest dobrze
			krawedzie.emplace_back(index_wierzcholka_z, index_kubelka_z, index_wierzcholka_do, index_kubelka_do);
		}
	}

	grafowe::krawedz& dodaj_krawedz(ID_W index_wierzcholka_z, ID_W index_wierzcholka_do) {
		// znajduje najmniejszy index kube³ka mo¿liwy do podpiêcia i dodaje tam krawedŸ
		ID_K index_kubelka_z = wierzcholki[index_wierzcholka_z].gdzie_mozna_podpiac_z();
		ID_K index_kubelka_do = wierzcholki[index_wierzcholka_do].gdzie_mozna_podpiac_do();
		ASSERT_Z_ERROR_MSG(index_kubelka_z != (ID_K)-1, "Nie ma miejsca z wierzcholka o indeksie: %d i opisie: %s\n" SEP index_wierzcholka_z SEP wierzcholki[index_wierzcholka_z].opis.c_str());
		ASSERT_Z_ERROR_MSG(index_kubelka_do != (ID_K)-1, "Nie ma miejsca do wierzcholka o indeksie: %d i opisie: %s\n" SEP index_wierzcholka_do SEP wierzcholki[index_wierzcholka_do].opis.c_str());
		dodaj_krawedz(index_wierzcholka_z, index_kubelka_z, index_wierzcholka_do, index_kubelka_do);
		return krawedzie.back(); // mo¿na przeczytaæ jakie indeksy zosta³y znalezione
	}

	grafowe::krawedz& dodaj_krawedz_nieskier(ID_W index_wierzcholka_A, ID_W index_wierzcholka_B) {
			   dodaj_krawedz(index_wierzcholka_A, index_wierzcholka_B);
		return dodaj_krawedz(index_wierzcholka_B, index_wierzcholka_A);
	}

	void dodaj_krawedz_nieskier(ID_W index_wierzcholka_A, ID_K  index_kubelka_A,
								ID_W index_wierzcholka_B, ID_K  index_kubelka_B) {
		dodaj_krawedz(index_wierzcholka_A, index_kubelka_A, index_wierzcholka_B, index_kubelka_B);
		dodaj_krawedz(index_wierzcholka_B, index_kubelka_B, index_wierzcholka_A, index_kubelka_A);
	}

	graf& operator+=(const graf& B) {
		//³¹czenie grafów, po zakoñczeniu graf nie jest spójny
		graf& A = *this; // nazwanie this A

		ID_W offset = A.liczba_wierzcholkow(); // przyda sie za chwile

		//A.wierzcholki.reserve(A.liczba_wierzcholkow()+B.liczba_wierzcholkow()); jest m¹drzejsze samo
		for(const auto& Bw : B.wierzcholki){
			A.wierzcholki.push_back(Bw);
		}

		//A.krawedzie.reserve(A.liczba_krawedzi() + B.liczba_krawedzi()); jest m¹drzejsze samo
		for (auto& Bk : B.krawedzie) {
			A.krawedzie.emplace_back(grafowe::krawedz(Bk.index_wierzcholka_z + offset, Bk.index_kubelka_z, Bk.index_wierzcholka_do + offset, Bk.index_kubelka_do));
		}

		return A;
	}

	graf operator+(const graf& B) const{
		//³¹czenie grafów, po zakoñczeniu graf nie jest spójny
		const graf& A = *this; // nazwanie this A
		graf R(A);
		return R += B;
	}

	bool czy_gotowy() const{
		bool ret = true;
		for(auto& w : wierzcholki){
			ret &= w.czy_wszystko_polaczone();
		}
		return ret;
	}
};

// czy to jest u¿yteczne?
constexpr bool Z_NAZWAMI = true;
constexpr bool BEZ_NAZW = false;

graf graf_lini(uint32_t liczba_wierzcholkow, bool z_nazwami = Z_NAZWAMI);
graf graf_lini_cykl(uint32_t liczba_wierzcholkow, bool z_nazwami = Z_NAZWAMI);
graf graf_krata_2D(uint32_t liczba_wierzcholkow_boku, bool z_nazwami = Z_NAZWAMI);
graf graf_krata_2D_z_przekatnymi(uint32_t liczba_wierzcholkow_boku, bool z_nazwami = Z_NAZWAMI);
graf graf_krata_3D(uint32_t liczba_wierzcholkow_boku, bool z_nazwami = Z_NAZWAMI);
void test_funkcji_tworzacych_grafy();

