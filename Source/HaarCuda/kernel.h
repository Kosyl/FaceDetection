#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "cuda_runtime.h"
#include "cuda.h"
#include "TypeDef.h"
#include "HaarArea.h"
#include "helper_cuda.h"

//suma pikseli w obrazie
__device__ long getSumInRect(UInt left, UInt top, UInt windowWidth, UInt windowHeight, UInt* values, int stride);

void launchHaarKernel(UInt* image_dev, float* weights_dev, bool* votingArea_dev, UInt imageStride, UInt weightsStride, HaarArea* allAreas_dev, UInt areasInStage[STAGES_COUNT], float thresholds[STAGES_COUNT], dim3 blockDim, dim3 gridDim);

__global__ void haarKernel(UInt* image_dev, float* weights_dev, bool* votingArea_dev, UInt imageStride, UInt weightsStride, HaarArea* allAreas_dev);

#endif