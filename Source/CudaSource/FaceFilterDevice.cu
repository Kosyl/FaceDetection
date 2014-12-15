#include "FaceFilterDevice.h"
#include <cmath>


__global__ void MaskKernel(unsigned char *img, unsigned char *mask, int sizeX, int sizeY) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < sizeX * sizeY)
	{
		if (mask[i] == 0)
			img[i] = 0;
	}
}

__global__ void ErodeKernel(unsigned char *img, unsigned char *out, int eltSize, int sizeX, int sizeY) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < sizeX * sizeY)
	{
		int xPos = index % sizeX;
		int yPos = (int) index / sizeX;
		const int factor = eltSize / 2;

		if (xPos < eltSize || xPos > sizeX - eltSize || yPos < eltSize || yPos > sizeY - eltSize) 
		{
			out[index] = 0;
		}
		else
		{
			int val = 255;

			for(int i = -factor; i < factor; i++ )
			{
				for(int j = -factor; j < factor; j++ )
				{
					if( img[xPos + i + (sizeX * (yPos + j) )] == 0) 
					{
						val = 0;
					}
				}
			}

			for(int i = -factor; i < factor; i++ ) 
			{
				for(int j = -factor; j < factor; j++ ) 
				{
					out[xPos + i + (sizeX * (yPos + j) )] = val;
				}
			}
		}
	}
}

__global__ void DilateKernel(unsigned char *img, unsigned char *out, int eltSize, int sizeX, int sizeY) {
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < sizeX * sizeY)
	{
		int xPos = index % sizeX;
		int yPos = (int) index / sizeX;
		const int factor = eltSize / 2;

		if (xPos < eltSize || xPos > sizeX - eltSize || yPos < eltSize || yPos > sizeY - eltSize)
		{
			
			out[index] = 0;
		}
		else
		{
			int val = 0;

			for(int i = -factor; i < factor; i++ )
			{
				for(int j = -factor; j < factor; j++ )
				{
					val += img[xPos + i + (sizeX * (yPos + j) )];
				}
			}

			if (val)
				val = 255;

			for(int i = -factor; i < factor; i++ )
			{
				for(int j = -factor; j < factor; j++ )
				{
					out[xPos + i + (sizeX * (yPos + j) )] = val;
				}
			}
		}
	}
}

/// wartoœæ musi byæ taka sama, jak liczba w¹tków
#define TCNT 512

__global__ void FindMaxMinKernel(unsigned char *in, unsigned char *out, int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	__shared__ unsigned char temp[512];

	if( index < size)
	{
		temp[tid] = in[index];

		__syncthreads();

		for (unsigned int s=blockDim.x/2; s > 0; s>>=1)
		{
			if (tid < s) 
			{
				if( temp[tid] > temp[tid + s] )
					temp[tid] = temp[tid];
				else
					temp[tid] = temp[tid + s];
			}
			__syncthreads();
		}
	}

	if (tid == 0)
	{
        out[blockIdx.x] = temp[0];
    }
}

__global__ void StretchKernel(unsigned char *img, unsigned char *maxVal, int sizeX, int sizeY )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < sizeX * sizeY)
	{
		const unsigned char newVal = (unsigned char)(( (float)img[index] / (float)maxVal[0]) * 255.f);

		if (newVal > 95 && newVal < 240)
			img[index] = newVal;
		else
			img[index] = 0;
	}
}



#if 0
 __global__ void CloseKernel(unsigned char *img, unsigned char *out, int *elt, int eltSize, int sizeX, int sizeY) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < sizeX * sizeY)
	{
		int xPos = i % sizeX;
		int yPos = (int) i / sizeX;
		const int factor = eltSize / 2;
		
		uchar *tempImg;
		// structElt = new int[2 * eltSize * eltSize];
		// tempImg = new uchar[sizeX * sizeY];

		// CreateStructElt(structElt, eltSize);

		//dilate
 		if (i < sizeX * sizeY) {
				if (xPos < eltSize || xPos > sizeX - eltSize || yPos < eltSize || yPos > sizeY - eltSize)
				{
					tempImg[i] = 0;
				}
				else
				{
					int val = 0;

					for(int i = -factor; i < factor; i++ )
					{
						for(int j = -factor; j < factor; j++ )
						{
							val += img[xPos + i + (sizeX * (yPos) )];
						}
					}

					if (val)
						val = 255;

					for(int i = -factor; i < factor; i++ )
					{
						for(int j = -factor; j < factor; j++ )
						{
							tempImg[xPos + factor + (sizeX * (j + factor)) ] = val;
						}
					}
				}

		//erode
		
				unsigned char val = 0;

				if (xPos < eltSize || xPos > sizeX - eltSize || yPos < eltSize || yPos > sizeY - eltSize)
				{
					out[i] = 0;
				}
				else
				{
					unsigned char val = 0;
					
					for(int i = -factor; i < factor; i++ )
					{
						for(int j = -factor; j < factor; j++ )
						{
							if (tempImg[xPos + i + (sizeX * (yPos + j))] == 0)
							{
								val = 0;
								break;
							}
						}
					}

					// val = 255;
				}

				for(int i = -factor; i < factor; i++ )
				{
					for(int j = -factor; j < factor; j++ )
					{
						out[xPos + i + (sizeX * (yPos + j))] = val;
					}
				}
			}
		}
	}
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
#endif