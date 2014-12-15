/// Declarations of functions, which call CUDA Kernels

#ifndef _FACE_FILTER_WRAP_H_
#define _FACE_FILTER_WRAP_H_

#include "cuda_runtime.h"
#include "cuda.h"
#include <vector>

/*void RGB2IRBDevice( unsigned char *&imgIn, float *&imgOut, int sizeX, int sizeY );
void MedianFilterDevice( float *imgIn, float *imgOut, int scale, int *ngb, int sizeX, int sizeY );
void HueSaturationDevice( float *hue, float *saturation, float *By, float *Rb, int sizeX, int sizeY );
void TextureDevice( float *texture, float *imgIn, float *imgFiltered, int sizeX, int sizeY );
void GenerateMapDevice( float *medianImg, float *hue, float *saturation, unsigned char *map, int sizeX, int sizeY );
*/

void CloseDevice(unsigned char *img, unsigned char *out, int eltSize, int sizeX, int sizeY);
void MaskDevice(unsigned char *img, unsigned char *mask, int sizeX, int sizeY);
void StretchDevice(unsigned char *img, int sizeX, int sizeY);

/// TODO - labelowanie obszar�w
std::vector<HaarRectangle> FindFacesDevice(unsigned char *&img, unsigned char *&out, int sizeX, int sizeY);

#endif
