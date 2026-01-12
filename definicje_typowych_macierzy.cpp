#include "transformaty.h"

#include "definicje_typowych_macierzy.h"

//					         HADAMARD 2x2
zesp dane_hadamard[4] = { 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0),
						  1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0) };
const transformata_macierz<zesp> H(2, dane_hadamard);

//							   PAULI X
zesp dane_permutacja[4] = { 0.0, 1.0,
							1.0, 0.0 };
const transformata_macierz<zesp> X(2, dane_permutacja);

//							   PAULI Z
zesp dane_pauli_z[4] = { 1.0, 0.0,
						 0.0, -1.0 };
const transformata_macierz<zesp> Z(2, dane_pauli_z);

//							   PAULI Y
zesp dane_pauli_y[4] = { 0.0, zesp(0.0, -1.0),
						 zesp(0.0, 1.0), 0.0 };
const transformata_macierz<zesp> Y(2, dane_pauli_y);

//							IDENTYCZNOSCI
const transformata_macierz<zesp> I_1(1.0);
//-------------------------------------------------------------
zesp dane_I_2[4] = { 1.0, 0.0,
					 0.0, 1.0 };
const transformata_macierz<zesp> I_2(2, dane_I_2);
//-------------------------------------------------------------
zesp dane_I_3[9] = { 1.0, 0.0, 0.0,
					 0.0, 1.0, 0.0,
					 0.0, 0.0, 1.0 };
const transformata_macierz<zesp> I_3(3, dane_I_3);
//-------------------------------------------------------------
zesp dane_I_4[16] = { 1.0, 0.0, 0.0, 0.0,
					  0.0, 1.0, 0.0, 0.0,
					  0.0, 0.0, 1.0, 0.0,
					  0.0, 0.0, 0.0, 1.0 };
const transformata_macierz<zesp> I_4(4, dane_I_4);

//					 HADAMARD TENSOR HADAMARD
zesp dane_HxH[16] = { 0.5,  0.5,  0.5,  0.5,
					  0.5, -0.5,  0.5, -0.5,
					  0.5,  0.5, -0.5, -0.5,
					  0.5, -0.5, -0.5,  0.5 };
const transformata_macierz<zesp> HxH(4, dane_HxH);

//					        I TENSOR X
zesp dane_IxX[16] = { 0.0, 1.0, 0.0, 0.0,
					  1.0, 0.0, 0.0, 0.0,
					  0.0, 0.0, 0.0, 1.0,
					  0.0, 0.0, 1.0, 0.0 };
const transformata_macierz<zesp> IxX(4, dane_IxX);

//					        Kolejnosc Filipa
zesp dane_F_kolejnosc[16] = { 0.0, 0.0, 1.0, 0.0,
							 0.0, 1.0, 0.0, 0.0,
							 1.0, 0.0, 0.0, 0.0,
							 0.0, 0.0, 0.0, 1.0 };
const transformata_macierz<zesp> FK(4, dane_F_kolejnosc);

const transformata_macierz<zesp> TJF(mnoz(std_kierunki_krata, FK));

//					        Fourier 4x4
double sqrt_4 = sqrt(4.0);
zesp dane_Fourier_4[16] = { zesp(1.0, 0.0) / sqrt_4, zesp(1.0, 0.0)  / sqrt_4, zesp(1.0, 0.0)  / sqrt_4, zesp(1.0, 0.0)  / sqrt_4,
							zesp(1.0, 0.0) / sqrt_4, zesp(0.0, -1.0) / sqrt_4, zesp(-1.0, 0.0) / sqrt_4, zesp(0.0, 1.0)  / sqrt_4,
							zesp(1.0, 0.0) / sqrt_4, zesp(-1.0, 0.0) / sqrt_4, zesp(1.0, 0.0)  / sqrt_4, zesp(-1.0, 0.0) / sqrt_4,
							zesp(1.0, 0.0) / sqrt_4, zesp(0.0, 1.0)  / sqrt_4, zesp(-1.0, 0.0) / sqrt_4, zesp(0.0, -1.0) / sqrt_4 };
const transformata_macierz<zesp> Fourier_4(4, dane_Fourier_4);