/// Declarations of functions, which call CUDA Kernels

#ifndef _SKIN_FILTER_WRAP_H_
#define _SKIN_FILTER_WRAP_H_

#include "cuda_runtime.h"
#include "cuda.h"

void RGB2IRBDevice( unsigned char *&imgIn, float *&imgOut, int sizeX, int sizeY );
void MedianFilterDevice( float *imgIn, float *imgOut, int scale, int *ngb, int sizeX, int sizeY );
void HueSaturationDevice( float *hue, float *saturation, float *By, float *Rb, int sizeX, int sizeY );
void TextureDevice( float *texture, float *imgIn, float *imgFiltered, int sizeX, int sizeY );
void GenerateMapDevice( float *medianImg, float *hue, float *saturation, unsigned char *map, int sizeX, int sizeY );

#endif