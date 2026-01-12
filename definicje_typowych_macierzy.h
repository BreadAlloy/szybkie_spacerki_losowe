#pragma once

#include "transformaty.h"

//					         HADAMARD 2x2
extern const transformata_macierz<zesp> H;
#define hadamard H

//							   PAULI X
extern const transformata_macierz<zesp> X;
#define pauliX X

//							   PAULI Z
extern const transformata_macierz<zesp> Z;
#define pauliZ Z

//							   PAULI Y
extern const transformata_macierz<zesp> Y;
#define pauliY Y

//							IDENTYCZNOSCI
extern const transformata_macierz<zesp> I_1;
//-------------------------------------------------------------
extern const transformata_macierz<zesp> I_2;
#define pauliI I_2
#define identycznosc_2 I_2
//-------------------------------------------------------------
extern const transformata_macierz<zesp> I_3;
#define identycznosc_3 I_3
//-------------------------------------------------------------
extern const transformata_macierz<zesp> I_4;
#define identycznosc_4 I_4

//					 HADAMARD TENSOR HADAMARD
extern const transformata_macierz<zesp> HxH;
#define hadamardxhadamard HxH

//					        I TENSOR X
extern const transformata_macierz<zesp> IxX;
#define pauliIxpauliX IxX
#define std_kierunki_krata IxX

//					        Kolejnosc Filipa
extern const transformata_macierz<zesp> FK;
#define FilipKolejnsc FK
extern const transformata_macierz<zesp> TJF;

//					        Fourier 4x4
extern const transformata_macierz<zesp> Fourier_4;
