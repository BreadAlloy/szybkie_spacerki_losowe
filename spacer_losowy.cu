#include "spacer_losowy.h"

#include "transformaty_wyspecializowane.h"

//#include "zesp.h"

__HD__ void testy_macierzy() {
	transformata_macierz<double> t1(1.0);
	transformata_macierz<double> t2(0.75, 0.5, 0.25, 0.5);
	transformata_macierz<double> t3((uint8_t)6);
	double temp[4] = { 0.5, 0.75, 0.5, 0.25 };
	transformata_macierz<double> t4((uint8_t)2, temp);
	transformata_macierz<double> t5(t2);
	transformata_macierz<double> t6 = t3;
	transformata_macierz<double> t7(mnoz(t4, t5));
	transformata_macierz<double> t8(tensor(t4, t5));
	IF_HOST(printf("t8: %s\n", t8.str().c_str());)
	bool t9 = (t6 == t3);
}
__device__ void cuda_test2() {
	printf("testuje2\n");
	testy_macierzy();
}

__global__ void cuda_test(){
	printf("testuje\n");
	testy_macierzy();
	cuda_test2();
}

__host__ void cuda_tester(){
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	cudaStream_t stream; // stream jest konieczny do printowania
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	cuda_test<<<16, 1, 0,stream>>>();

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	checkCudaErrors(cudaStreamSynchronize(stream));

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
}

#ifdef __CUDA_ARCH__
#include "transformaty.cu"
#include "zesp.cu"
#endif

