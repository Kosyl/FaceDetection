#include "SkinFilterDevice.h"
#include <cmath>

#define L(x) 105 * log( x + 1 )

__global__ void RGB2IRBKernel( unsigned char *imgIn, float *imgOut, int sizeX, int sizeY )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if( index < sizeX * sizeY )
	{
		const int stride = sizeX * sizeY;
	
		float R = float( imgIn[3 * index    ] );
		float G = float( imgIn[3 * index + 1] );
		float B = float( imgIn[3 * index + 2] );

		imgOut[              index ] = ( L(R) + L(B) + L(G) ) / 3;
		imgOut[ stride     + index ] = L(R) - L(G);
		imgOut[ 2 * stride + index ] = L(B) - ( L(G) + L(R) ) / 2;
	}
}

__global__ void MedianFilterKernel( float *imgIn, float *imgOut, int scale, int *ngb, int sizeX, int sizeY )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if( index < sizeX * sizeY )
	{
		float window[1000];
		
		const int xPos = index % sizeX;
		const int yPos = (int) index / sizeX;
		
		if( xPos < scale || yPos < scale || xPos > sizeX - scale || yPos > sizeY - scale )
		{
			imgOut[index] = imgIn[index];
			imgOut[index] = 255;
		}
		else
		{
			int windCnt = 0;

			for( int k = 0; k < 2 * scale * scale; k += 2 )
			{
				const float val = imgIn[ xPos + ngb[k] + (sizeX * (yPos + ngb[k+1])) ];
				window[windCnt] = val;
				windCnt++;
			}
			
			float median = GetMedianDevice( window, scale );
			imgOut[ index ] = median;
		}
	}
}

__device__ float GetMedianDevice( float *img, int scale )
{
	for( int x=0; x < scale * scale; x++ )
	{
		for( int y = 0; y < scale; y++ )
		{
			if(img[y] > img[y+1])
			{
				float temp = img[y+1];
				img[y+1] = img[y];
				img[y] = temp;
			}
		}
	}

	return img[ (((scale * scale) - 1 ) / 2 )+ 1 ];
}

__global__ void HueSaturationKernel( float *hue, float *saturation, float *By, float *Rb, int sizeX, int sizeY )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if( index < sizeX * sizeY )
	{
		const float  Rg_ = By[index];
		const float  By_ = Rb[index];

		hue[index] = (atan2f(Rg_, By_) / 3.14) * 180;
		saturation[index] = hypotf(Rg_, By_);
	}
}

void TextureKernel( float *texture, float *imgIn, float *imgFiltered, int sizeX, int sizeY )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if( index < sizeX * sizeY )
	{
		texture[index] = abs( imgIn[index] - imgFiltered[index] );
	}
}

void GenerateMapKernel( float *medianImg, float *hue, float *saturation, unsigned char *map, int sizeX, int sizeY )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if( index < sizeX * sizeY )
	{
	
		bool m = medianImg[index] < 4.5;
		bool h = hue[index] > 120 && hue[index] < 160;
		bool s = saturation[index] > 10 && saturation[index] < 60;

		if( m && h && s )
		{
			map[index] = 255;
			return;
		}

		m = medianImg[index] < 4.5;
		h = hue[index] > 150 && hue[index] < 180;
		s = saturation[index] > 20 && saturation[index] < 80;

		if( m && h && s )
		{
			map[index] = 255;
			return;
		}

		map[index] = 0;
	}
}

