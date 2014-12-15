#include "cuda_runtime.h"
#include "cuda.h"

#ifndef _FACE_FILTER_DEVICE_H_
#define _FACE_FILTER_DEVICE_H_
/*
__global__ void RGB2IRBKernel( unsigned char *imgIn, float *imgOut, int sizeX, int sizeY );
__global__ void MedianFilterKernel( float *imgIn, float *imgOut, int scale, int *ngb, int sizeX, int sizeY );
__global__ void HueSaturationKernel( float *hue, float *saturation, float *By, float *Rb, int sizeX, int sizeY );
__global__ void TextureKernel( float *texture, float *imgIn, float *imgFiltered, int sizeX, int sizeY );
__global__ void GenerateMapKernel( float *medianImg, float *hue, float *saturation, unsigned char *map, int sizeX, int sizeY );

__device__ float GetMedianDevice( float *img, int scale );
*/

__global__ void MaskKernel(unsigned char *img, unsigned char *mask, int sizeX, int sizeY);
__global__ void CloseKernel(unsigned char *img, unsigned char *out, int *elt, int eltSize, int sizeX, int sizeY);
__global__ void CreateStructEltKernel(int *elt, int eltSize);
__global__ std::vector<HaarRectangle> FindFacesKernel(unsigned char *img, unsigned char *out, int sizeX, int sizeY);

__device__ void Stretch_Color(unsigned char *img, int sizeX, int sizeY);

#endif
