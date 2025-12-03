#pragma once

#include "pomocne_funkcje.h"

#include <memory>

template <typename typ_wskaznika>
struct statyczny_wektor{ // wektor automatycznie mallocujacy i zwalnijacy, nie ma resize
	typ_wskaznika* pamiec_host = nullptr; // na device to pole jest nieu¿ywane
	typ_wskaznika* pamiec_device = nullptr;
	size_t rozmiar = 0;

	#define PAMIEC IF_HD(pamiec_host, pamiec_device)

	__HD__ statyczny_wektor(typ_wskaznika* pamiec, size_t rozmiar)
		: PAMIEC(pamiec), rozmiar(rozmiar) {
	}

	__HD__ statyczny_wektor(size_t poczatkowy_rozmiar = 0)
		: PAMIEC(nullptr), rozmiar(0) {
		if (poczatkowy_rozmiar != 0) this->malloc(poczatkowy_rozmiar);
	}

	__HD__ void malloc(size_t nowy_rozmiar) { // rozmiar jest w iloœci obiektów typu wskaŸnika
		ASSERT_Z_ERROR_MSG(PAMIEC == nullptr, "Coœ jest w pamieci\n");
		rozmiar = nowy_rozmiar;
		PAMIEC = (typ_wskaznika*)(::malloc(bajt_rozmiar()));
		ASSERT_Z_ERROR_MSG(PAMIEC != nullptr, "malloc zawiodl\n");
	}

	__host__ void cuda_malloc(){
		ASSERT_Z_ERROR_MSG(pamiec_device == nullptr, "Coœ jest ju¿ na device\n");
		checkCudaErrors(cudaMalloc((void**)&pamiec_device, bajt_rozmiar()));
	}

	__host__ void cuda_zanies(cudaStream_t stream){
		ASSERT_Z_ERROR_MSG(pamiec_device != nullptr, "Nie ma pamieci na gpu\n");
		ASSERT_Z_ERROR_MSG(pamiec_host != nullptr, "Nie ma pamieci na hoscie\n");
		checkCudaErrors(cudaMemcpyAsync((void*)pamiec_device, (void*)pamiec_host, bajt_rozmiar(), cudaMemcpyHostToDevice, stream));
	}

	__host__ void cuda_przynies(cudaStream_t stream){
		ASSERT_Z_ERROR_MSG(pamiec_device != nullptr, "Nie ma pamieci na gpu\n");
		ASSERT_Z_ERROR_MSG(pamiec_host != nullptr, "Nie ma pamieci na hoscie\n");
		checkCudaErrors(cudaMemcpyAsync((void*)pamiec_host, (void*)pamiec_device, bajt_rozmiar(), cudaMemcpyDeviceToHost, stream));
	}

	__host__ void cuda_free() {
		ASSERT_Z_ERROR_MSG(pamiec_device != nullptr, "Nic nie ma w pamieci");
		checkCudaErrors(cudaFree((void*)pamiec_device));
		pamiec_device = nullptr;
	}

	__HD__ void memset(typ_wskaznika wartosc) {
		for (size_t i = 0; i < rozmiar; i++) operator[](i) = wartosc;
	}

	__HD__ size_t bajt_rozmiar() const {
		return rozmiar * sizeof(typ_wskaznika);
	}

	__HD__ void free() {
		ASSERT_Z_ERROR_MSG(PAMIEC != nullptr, "Nic nie ma w pamieci");
		::free(PAMIEC);
		PAMIEC = nullptr;
	}

	__HD__ statyczny_wektor(const statyczny_wektor& kopiowany) : PAMIEC(nullptr), rozmiar(0) {
		this->malloc(kopiowany.rozmiar);
		::memcpy(this->PAMIEC, kopiowany.PAMIEC, kopiowany.bajt_rozmiar());
	}

	__HD__ statyczny_wektor& operator=(const statyczny_wektor& kopiowany) {
		if(this->rozmiar != kopiowany.rozmiar){
			if (this->rozmiar != 0) this->free();
			this->malloc(kopiowany.rozmiar);
		}
		::memcpy(this->PAMIEC, kopiowany.PAMIEC, kopiowany.bajt_rozmiar());
		return *this;
	}

	__HD__ typ_wskaznika& operator[](uint64_t index) {
		lepszy_assert(index < rozmiar);
		return PAMIEC[index];
	}

	__HD__ typ_wskaznika operator[](uint64_t index) const {
		lepszy_assert(index < rozmiar);
		return PAMIEC[index];
	}

	__HD__ ~statyczny_wektor() {
		if (PAMIEC != nullptr) {
			//for(uint64_t i = 0; i < rozmiar; i++) (pamiec[i]).~typ_wskaznika();
			this->free();
		}
		IF_HOST(if(pamiec_device != nullptr) cuda_free();)
		rozmiar = 0;
	}
};

template <typename typ_wskaznika>
struct estetyczny_wektor { // po prostu otoczka dla wskaznika do pamieci
	typ_wskaznika* pamiec = nullptr;
	size_t rozmiar = 0;

	__HD__ estetyczny_wektor(typ_wskaznika* pamiec, size_t rozmiar)
		: pamiec(pamiec), rozmiar(rozmiar) {
	}

	__HD__ estetyczny_wektor(const statyczny_wektor<typ_wskaznika>& rozszerzany)
		: pamiec(rozszerzany.PAMIEC), rozmiar(rozszerzany.rozmiar) {
	}

	__HD__ typ_wskaznika& operator[](uint64_t index) {
		ASSERT_Z_ERROR_MSG(index < rozmiar, "Index nie jest w wektorze");
		return pamiec[index]; 
	}

	__HD__ typ_wskaznika operator[](uint64_t index) const {
		ASSERT_Z_ERROR_MSG(index < rozmiar, "Index nie jest w wektorze");
		return pamiec[index];
	}

	__HD__ estetyczny_wektor(const estetyczny_wektor& kopiowany)
		: pamiec(kopiowany.pamiec), rozmiar(kopiowany.rozmiar)
	{
	}

	__HD__ estetyczny_wektor& operator=(const estetyczny_wektor& kopiowany) {
		pamiec = kopiowany.pamiec;
		rozmiar = kopiowany.rozmiar;
		return *this;
	}
};

template<typename typ_wskaznika>
struct wektor_do_pushbackowania{
	typ_wskaznika* pamiec = nullptr;
	size_t rozmiar = 0;
	size_t rozmiar_zmallocowany = 0;

	__HD__ wektor_do_pushbackowania(size_t poczatkowy_rozmiar = 0)
		: pamiec(nullptr), rozmiar(0), rozmiar_zmallocowany(0){
		if (poczatkowy_rozmiar != 0) this->malloc(poczatkowy_rozmiar);
	}

	__HD__ void przebuduj(size_t poczatkowy_rozmiar){
		if(rozmiar_zmallocowany != poczatkowy_rozmiar){
			if (pamiec != nullptr) {
				this->free();
			}
			this->malloc(poczatkowy_rozmiar);
		} else {
			rozmiar = 0;
		}
	}

	//__HD__ wektor_do_pushbackowania& operator=(wektor_do_pushbackowania&& kopiowany) {
	//	if(pamiec != nullptr) this->free();
	//	pamiec = kopiowany.pamiec;
	//	rozmiar = kopiowany.rozmiar;
	//	rozmiar_zmallocowany = kopiowany.rozmiar_zmallocowany;
	//	return *this;
	//}

	__HD__ wektor_do_pushbackowania& operator=(const wektor_do_pushbackowania& kopiowany){
		if(pamiec != nullptr){
			this->free();
		}
		this->malloc(kopiowany.rozmiar_zmallocowany);
		::memcpy(pamiec, kopiowany.pamiec, kopiowany.rozmiar);
		return *this;
	} 

	__HD__ void malloc(size_t nowy_rozmiar) { // rozmiar jest w iloœci obiektów typu wskaŸnika
		ASSERT_Z_ERROR_MSG(pamiec == nullptr, "Coœ jest w pamieci");
		rozmiar = 0;
		rozmiar_zmallocowany = nowy_rozmiar;
		pamiec = (typ_wskaznika*)(::malloc(bajt_rozmiar()));
		ASSERT_Z_ERROR_MSG(pamiec != nullptr, "malloc zawiodl");
	}

	__HD__ void memset(typ_wskaznika wartosc) {
		for (size_t i = 0; i < rozmiar; i++) pamiec[i] = wartosc;
	}

	__HD__ size_t bajt_rozmiar() const {
		return rozmiar_zmallocowany * sizeof(typ_wskaznika);
	}

	__HD__ void free() {
		ASSERT_Z_ERROR_MSG(pamiec != nullptr, "Nic nie ma w pamieci");
		::free(pamiec);
		rozmiar_zmallocowany = 0;
		rozmiar = 0;
		pamiec = nullptr;
	}

	__HD__ void pushback(typ_wskaznika nowe){
		ASSERT_Z_ERROR_MSG(rozmiar < rozmiar_zmallocowany, "Nie ma juz miejsca w wektorze");
		pamiec[rozmiar] = nowe;
		rozmiar++;
	}

	__HD__ typ_wskaznika& operator[](uint64_t index) {
		lepszy_assert(index < rozmiar);
		return pamiec[index];
	}

	__HD__ typ_wskaznika operator[](uint64_t index) const {
		lepszy_assert(index < rozmiar);
		return pamiec[index];
	}

	__HD__ ~wektor_do_pushbackowania() {
		if (pamiec != nullptr) {
			//for(uint64_t i = 0; i < rozmiar; i++) (pamiec[i]).~typ_wskaznika();
			this->free();
		}
	}
};


