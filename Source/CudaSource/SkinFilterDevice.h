#include "cuda_runtime.h"
#include "cuda.h"

#ifndef _SKIN_FILTER_DEVICE_H_
#define _SKIN_FILTER_DEVICE_H_

__global__ void RGB2IRBKernel( unsigned char *imgIn, float *imgOut, int sizeX, int sizeY );
__global__ void MedianFilterKernel( float *imgIn, float *imgOut, int scale, int *ngb, int sizeX, int sizeY );
__global__ void HueSaturationKernel( float *hue, float *saturation, float *By, float *Rb, int sizeX, int sizeY );
__global__ void TextureKernel( float *texture, float *imgIn, float *imgFiltered, int sizeX, int sizeY );
__global__ void GenerateMapKernel( float *medianImg, float *hue, float *saturation, unsigned char *map, int sizeX, int sizeY );

__device__ float GetMedianDevice( float *img, int scale );

#endif