#pragma once

#include "pomocne_funkcje.h"

constexpr uint32_t porcja_czytania = 4 * 1024;

#define ZMIENNE		char* poczatek_wyrazenia = buffer;\
					char* biezaca_pozycja = buffer;

#define ZLAP_WYRAZENIE 		while(*biezaca_pozycja != ','){\
									biezaca_pozycja++;\
								}\
								*biezaca_pozycja = NULL;\

#define KOLEJNE_WYRAZENIE	*biezaca_pozycja = ',';\
								biezaca_pozycja++;\
								poczatek_wyrazenia = biezaca_pozycja;

namespace zapamietywacz {

	//struct baza_operatorow;
	//struct baza_krawedzi;
	//struct baza_konfiguracji;

	struct baza_baza {
		FILE* fd = nullptr;

		void czytaj_plik(std::string sciezka_operatory) {
			fd = fopen(sciezka_operatory.c_str(), "a+b");
			ASSERT_Z_ERROR_MSG(fd != NULL, "Cos nie tak z plikiem\n");
			char buffer[porcja_czytania];
			while (nullptr != std::fgets(buffer, porcja_czytania, fd)) {
				czytaj_linie(buffer);
			}
		}

		virtual void czytaj_linie(char* buffer) = 0;

		~baza_baza() {
			fclose(fd);
		}
	};

	typedef std::vector<uint32_t> uklad_operatorow;

	template<typename transformata>
	struct baza_transformat : baza_baza {
		std::vector<transformata> transformaty;

		__host__ baza_transformat(std::string folder) {
			czytaj_plik(folder + "\\transformaty.csv");
		}

		__host__ uint32_t dodaj_transformate(const transformata& nowa, bool sprawdz_czy_duplikat = true) {
			uint32_t ile_do_przejrzenia = (uint32_t)transformaty.size();
			if(sprawdz_czy_duplikat){
				for (uint32_t i = 0; i < ile_do_przejrzenia; i++) {
					if (transformaty[i] == nowa) return i;
				}
			}
			uint32_t id = zapisz_linie(nowa);
			transformaty.push_back(nowa);
			return id;
		}

		__host__ void czytaj_linie(char* buffer) override {
			ZMIENNE

				ZLAP_WYRAZENIE

				uint32_t id;
			id = std::stoi(poczatek_wyrazenia);

			KOLEJNE_WYRAZENIE
				ZLAP_WYRAZENIE

				uint8_t arrnosc;
			arrnosc = std::stoi(poczatek_wyrazenia);

			transformata czytana(arrnosc);
			for (uint8_t i = 0; i < arrnosc; i++) {
				for (uint8_t j = 0; j < arrnosc; j++) {
					KOLEJNE_WYRAZENIE
						ZLAP_WYRAZENIE
						z_bin(poczatek_wyrazenia, czytana(i, j));
				}
			}
			ASSERT_Z_ERROR_MSG(*(biezaca_pozycja + 1) == '\n', "Nie pamietam\n");
			ASSERT_Z_ERROR_MSG(czytana.sprawdz(), "Nie pamietam\n");
			transformaty.push_back(czytana);
		}

		__host__ uint64_t zapisz_linie(const transformata& nowa) {
			fseek(fd, 0, SEEK_END);
			uint64_t id_nowego = transformaty.size();
			fprintf(fd, "%d,", id_nowego);
			fprintf(fd, "%d,", nowa.arrnosc);
			std::string temp;
			for (uint8_t i = 0; i < nowa.arrnosc; i++) {
				for (uint8_t j = 0; j < nowa.arrnosc; j++) {
					temp = do_bin(nowa(i, j));
					fprintf(fd, "%s,", temp.c_str());
				}
			}
			fprintf(fd, "\n");
			return id_nowego;
		}
	};
}
