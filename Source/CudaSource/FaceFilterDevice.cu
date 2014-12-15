#include "FaceFilterDevice.cuh"
#include <cmath>


__global__ void MaskKernel(unsigned char *img, unsigned char *mask, int sizeX, int sizeY) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < sizeX * sizeY)
	{
		if (mask[i] == 0)
			img[i] = 0;
	}
}

 __global__ void CloseKernel(unsigned char *img, unsigned char *out, int *elt, int eltSize, int sizeX, int sizeY) {
	 /* tu zmienilem tylko kawawlek */
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int xPos = i % sizeX;
	int yPos = (int) i / sizeX;
	const int eltSize = 7;

	int   *structElt;
	uchar *tempImg;
	structElt = new int[2 * eltSize * eltSize];
	tempImg = new uchar[sizeX * sizeY];

	CreateStructElt(structElt, eltSize);

	//dilate
 	if (i < sizeX * sizeY) {
			if (xPos < eltSize || xPos > sizeX - eltSize || yPos < eltSize || yPos > sizeY - eltSize)
			{
				tempImg[i + sizeX * j] = 0;
			}
			else
			{
				int val = 0;

				for (int k = 0; k < 2 * eltSize * eltSize; k += 2)
				{
					val += img[i + structElt[k] + (sizeX * (j + structElt[k + 1]))];
				}

				if (val)
					val = 255;

				for (int k = 0; k < 2 * eltSize * eltSize; k += 2)
				{
					tempImg[i + structElt[k] + (sizeX * (j + structElt[k + 1]))] = val;
				}
			}
		}

	//erode
	for (int i = 0; i < sizeX; i++)
	{
		for (int j = 0; j < sizeY; j++)
		{
			uchar val = 0;

			if (i < eltSize || i > sizeX - eltSize || j < eltSize || j > sizeY - eltSize)
			{
				out[i + sizeX * j] = 0;
			}
			else
			{
				for (int k = 0; k < 2 * eltSize * eltSize; k += 2)
				{
					if (tempImg[i + structElt[k] + (sizeX * (j + structElt[k + 1]))] == 0)
					{
						val = 0;
						break;
					}

					val = 255;
				}

				for (int k = 0; k < 2 * eltSize * eltSize; k += 2)
				{
					out[i + structElt[k] + (sizeX * (j + structElt[k + 1]))] = val;
				}
			}
		}
	}

	delete[] structElt;
	delete[] tempImg;
}
__global__ void CreateStructEltKernel(int *elt, int eltSize) {
	int factor = (eltSize - 1) / 2;
	int currPos = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
		int xPos = i % eltSize;
	int yPos = (int) i / eltSize;
	std::cout << "Generating ngb: \n";

	if (i < elt_size * elt_size) {
		elt[2 * i] = xPos;
		elt[2 * i + 1] = yPos;
	}
}

__device__ void StretchColor(unsigned char *img, int sizeX, int sizeY)
{
	uchar maxVal = 0;
	// uchar minVal = 255;

	for (int i = 0; i < sizeX * sizeY; i++)
	{
		const int pxlVal = (int)img[i];

		/*if( pxlVal < minVal )
			minVal = pxlVal;*/

		if (pxlVal > maxVal)
			maxVal = pxlVal;
	}

	for (int i = 0; i < sizeX * sizeY; i++)
	{
		const int newVal = (((float)img[i] / (float)maxVal) * 255.f);

		if (newVal > 95 && newVal < 240)
			img[i] = newVal;
		else
			img[i] = 0;
	}

}


__global__ std::vector<HaarRectangle> FindFacesKernel(unsigned char *img, unsigned char *out, int sizeX, int sizeY) {
		
}
