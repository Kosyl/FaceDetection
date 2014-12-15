#include "FaceFilterWrap.h"
#include "FaceFilterDevice.cuh"
#include "stdio.h"

#define N 512

void MaskDevice(unsigned char *&img, unsigned char *&mask, const int sizeX, const int sizeY)
{
	const int blockCnt = ((sizeX * sizeY) / N) + 1;

	MaskKernel<<< blockCnt, N >>>(img, mask, sizeX, sizeY);

	printf( "Mask\n" );
}

void CreateStructEltDevice(int *&elt, int eltSize) {
	const int blockCnt = ((eltSize * eltSize) / N) + 1;

	CreateStructEltKernel<<< blockCnt, N >>>(elt, eltSize);

	printf( "Mask\n" );
}
void CloseDevice(unsigned char *&img, unsigned char *&out, int *&elt, int eltSize, int sizeX, int sizeY) {
	const int blockCnt = ((sizeX * sizeY) / N) + 1;

	CloseKernel<<< blockCnt, N >>>(img, out, elt, eltSize, sizeX, sizeY);

	printf( "Mask\n" );
}

std::vector<HaarRectangle> FindFacesDevice(unsigned char *&img, unsigned char *&out, int sizeX, int sizeY) {
	const int blockCnt = ((sizeX * sizeY) / N) + 1;

	FindFacesKernel<<< blockCnt, N >>>(img, out, sizeX, sizeY);

	printf( "Mask\n" );
}
