#include "SkinFilterWrap.h"
#include "SkinFilterDevice.h"
#include "stdio.h"

#define N 512

void RGB2IRBDevice( unsigned char *&imgIn, float *&imgOut, int sizeX, int sizeY )
{
	const int blockCnt = ((sizeX * sizeY) / N) + 1;

	RGB2IRBKernel<<< blockCnt, N >>>(imgIn, imgOut, sizeX, sizeY);
	printf( "Color Transform Kernel done\n" );
	
	cudaError err;
	err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		printf( "cudaCheckError() failed at : %s\n", cudaGetErrorString( err ) );
		getchar();
		// exit(0);
	}
}

void MedianFilterDevice( float *imgIn, float *imgOut, int scale, int *ngb, int sizeX, int sizeY )
{
	const int blockCnt = ((sizeX * sizeY) / N) + 1;
	MedianFilterKernel<<< blockCnt, N >>>( imgIn, imgOut, scale, ngb, sizeX, sizeY );
	printf( "Median filter Kernel done\n" );

	cudaError err;
	err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		printf( "cudaCheckError() failed at : %s\n", cudaGetErrorString( err ) );
		getchar();
	}
}

void HueSaturationDevice( float *hue, float *saturation, float *By, float *Rb, int sizeX, int sizeY )
{
	const int blockCnt = ((sizeX * sizeY) / N) + 1;
	HueSaturationKernel<<< blockCnt, N >>>(hue, saturation, By, Rb, sizeX, sizeY );
	printf( "Hue and Saturation calculated\n" );
	
	cudaError err;
	err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		printf( "cudaCheckError() failed at : %s\n", cudaGetErrorString( err ) );
		getchar();
	}
}

void TextureDevice( float *texture, float *imgIn, float *imgFiltered, int sizeX, int sizeY )
{
	const int blockCnt = ((sizeX * sizeY) / N) + 1;
	TextureKernel<<< blockCnt, N >>>(texture, imgIn, imgFiltered, sizeX, sizeY );
	printf( "Texture calculated\n" );

	cudaError err;
	err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		printf( "cudaCheckError() failed at : %s\n", cudaGetErrorString( err ) );
		getchar();
	}
}

void GenerateMapDevice( float *medianImg, float *hue, float *saturation, unsigned char *map, int sizeX, int sizeY )
{
	const int blockCnt = ((sizeX * sizeY) / N) + 1;
	GenerateMapKernel<<< blockCnt, N >>>(medianImg, hue, saturation, map, sizeX, sizeY);
	printf( "Map calculated\n" );

	cudaError err;
	err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		printf( "cudaCheckError() failed at : %s\n", cudaGetErrorString( err ) );
		getchar();
	}
}