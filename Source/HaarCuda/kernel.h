#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "cuda_runtime.h"
#include "cuda.h"
#include "TypeDef.h"
#include "HaarArea.h"
#include "helper_cuda.h"

//suma pikseli w obrazie
__device__ long getSumInRect(UInt left, UInt top, UInt windowWidth, UInt windowHeight, UInt* values, int stride);

//suma kwadratow pikseli w obrazie
__device__ long getSum2InRect(UInt left, UInt top, UInt windowWidth, UInt windowHeight, UInt* values2, int stride);

// wrapper kernela
// image_dev - obraz calkowy
// image2_dev - obraz calkowy kwadratow pikseli
// numPhases - ilosc badanych malych faz w duzej fazie (zaalokowanych w pamieci stalej)
// xStep/yStep - przesuniecie okna w poziomie/pionie
// xEnd/yEnd - max. polozenia okna
// stride - offset miedzy wierszami obrazow calkowych
// windowsSize - rozmiar okna (kwadratowe)
// invArea - odwrotnosc pola okna - uzywana do skalowania wynikow
// votingArea_dev - wskaznik na mape wynikow true/false, tzn "jest twarz/nie ma twarzy"
// threshold - prog wykrycia twarzy w danej duzej fazie
// areasStart - poczatek fragmentu tablicy malych faz; poczawszy od niego, numPhases malych faz zostanie skopiowanych do pamieci stalej
// blockDim, gridDim - wymiary siatki watkow
void launchHaarKernel(UInt* image_dev, UInt* image2_dev, int numPhases, int xStep, int xEnd, int yStep, int yEnd, int stride, int windowSize, float invArea, bool* votingArea_dev, float threshold, HaarArea* areasStart, dim3 blockDim, dim3 gridDim);

//kernel z rozwinieta petla po malych fazach
template<int K>
__global__ void haarKernelK(UInt* image_dev, UInt* image2_dev, int numPhases, int xStep, int xEnd, int yStep, int yEnd, int stride, int windowSize, float invArea, bool* votingArea_dev, float threshold);

//kernel z normalna petla
__global__ void haarKernel(UInt* image_dev, UInt* image2_dev, int numPhases, int xStep, int xEnd, int yStep, int yEnd, int stride, int windowSize, float invArea, bool* votingArea_dev, float threshold);

#endif