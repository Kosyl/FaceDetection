#include "FaceFilterWrap.h"
#include "FaceFilterDevice.h"
#include "stdio.h"

#define N 512

void CloseDevice(unsigned char *img, unsigned char *out, int eltSize, int sizeX, int sizeY) {
	const int blockCnt = ((sizeX * sizeY) / N) + 1;

	unsigned char* tempImg;
	cudaMalloc(&tempImg, sizeof(unsigned char) * sizeX * sizeY);

	DilateKernel<<<blockCnt, N>>>(img, tempImg, eltSize, sizeX, sizeY);
	ErodeKernel<<<blockCnt, N>>>(tempImg, out, eltSize, sizeX, sizeY);
	
	cudaFree(tempImg);

	printf( "Img Closed\n" );
}

void MaskDevice(unsigned char *img, unsigned char *mask, const int sizeX, const int sizeY)
{
	const int blockCnt = ((sizeX * sizeY) / N) + 1;

	MaskKernel<<< blockCnt, N >>>(img, mask, sizeX, sizeY);

	printf( "Mask image\n" );
}

void StretchDevice(unsigned char *img, int sizeX, int sizeY)
{
	int blockCnt = ((sizeX * sizeY) / N) + 1;
	int size = sizeX * sizeY;

	unsigned char* res;
	unsigned char* input;

	cudaMalloc(&res, sizeof(unsigned char) * blockCnt);
	cudaMalloc(&input, sizeof(unsigned char) * size);
	cudaMemcpy(input, img, sizeof(unsigned char) * size, cudaMemcpyDeviceToDevice);

	while(size > 0)
	{
		// printf( " size: %d blockCnt: %d\n", size, blockCnt );
		FindMaxMinKernel<<<blockCnt, N>>>(input, res, size);
		cudaMemcpy(res, input, sizeof(unsigned char) * blockCnt, cudaMemcpyDeviceToDevice);
		size /= 512;
		blockCnt = (blockCnt / 512) + 1;
	}

	StretchKernel<<<blockCnt, N>>>(img, &input[0], sizeX, sizeY);

	printf("Stretch color\n");
	
	cudaFree(res);
	cudaFree(input);
}

/*std::vector<HaarRectangle> FindFacesDevice(unsigned char *&img, unsigned char *&out, int sizeX, int sizeY) {
	const int blockCnt = ((sizeX * sizeY) / N) + 1;

	FindFacesKernel<<< blockCnt, N >>>(img, out, sizeX, sizeY);

	printf( "Mask\n" );
}*/
