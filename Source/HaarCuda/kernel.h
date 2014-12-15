#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "cuda_runtime.h"
#include "cuda.h"
#include "TypeDef.h"
#include "HaarArea.h"

__device__ long getSumInRect(UInt left, UInt top, UInt windowWidth, UInt windowHeight, UInt* values, int stride);

__device__ long getSum2InRect(UInt left, UInt top, UInt windowWidth, UInt windowHeight, UInt* values2, int stride);

__global__ void haarKernel(UInt* image_dev, UInt* image2_dev, float* stageSums_dev, HaarArea* areas_dev, int numPhases, int xStep, int xEnd, int yStep, int yEnd, int stride, int windowSize, float invArea);

void launchHaarKernel(UInt* image_dev, UInt* image2_dev, float* stageSums_dev, HaarArea* areas_dev, int numPhases, int xStep, int xEnd, int yStep, int yEnd, int stride, int windowSize, float invArea, dim3 blockDim, dim3 gridDim);

#endif