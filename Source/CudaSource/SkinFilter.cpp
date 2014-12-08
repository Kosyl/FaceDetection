
#include "SkinFilter.h"
#include <cmath>

#include <assert.h>
#include "SkinFilterWrap.h"
#include <stdio.h>

#define L(x) 105 * log( x + 1 )

SkinFilter::SkinFilter( void )
{
	m_ngb = NULL;
}

SkinFilter::~SkinFilter( void )
{
	if( m_ngb != NULL )
		delete[] m_ngb;
}

void SkinFilter::Filter( unsigned char *&img, unsigned char *&map, const int sizeX, const int sizeY )
{
	assert( sizeX );
	assert( sizeY );

	m_sizeX  = sizeX;
	m_sizeY  = sizeY;
	m_stride = m_sizeX * m_sizeY;

	const int orgScale = (m_sizeX + m_sizeY) / 320;
	int tmpScale       = 0;

	float *irbImageDevice;
	unsigned char *imgInputDevice;

	cudaMalloc(&imgInputDevice, m_stride * 3 * sizeof(unsigned char));
	cudaMalloc(&irbImageDevice, m_stride * 3 * sizeof(float));
	cudaMemcpy(imgInputDevice, img, m_stride * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	RGB2IRBDevice( imgInputDevice,irbImageDevice, m_sizeX, m_sizeY );
	
	cudaFree(imgInputDevice);
	
	tmpScale = CalcScale(orgScale, 8);
	
	int *ngbDevice;
	cudaMalloc(&ngbDevice, 2 * tmpScale * tmpScale * sizeof(int) );
	cudaMemcpy(ngbDevice, m_ngb, 2 * tmpScale * tmpScale * sizeof(int), cudaMemcpyHostToDevice);

	float *deviceMedianImg;
	cudaMalloc(&deviceMedianImg, m_stride * sizeof(float) );

	MedianFilterDevice( irbImageDevice, deviceMedianImg, tmpScale, ngbDevice, m_sizeX, m_sizeY );
	
	float *newRgDevice;
	float *newByDevice;
	cudaMalloc(&newRgDevice, m_stride * sizeof(float) );
	cudaMalloc(&newByDevice, m_stride * sizeof(float) );

	float *RgDevice = irbImageDevice + m_stride;
	float *ByDevice = irbImageDevice + 2 * m_stride;
	
	// tmpScale = CalcScale(orgScale, 4);
	MedianFilterDevice( RgDevice, newRgDevice, tmpScale, ngbDevice, m_sizeX, m_sizeY );
	MedianFilterDevice( ByDevice, newByDevice, tmpScale, ngbDevice, m_sizeX, m_sizeY );

	float *hueDevice;
	float *saturationDevice;
	float *textureDevice;

	cudaMalloc(&hueDevice, m_stride * sizeof(float) );
	cudaMalloc(&saturationDevice, m_stride * sizeof(float) );
	cudaMalloc(&textureDevice, m_stride * sizeof(float) );

	HueSaturationDevice( hueDevice, saturationDevice, newRgDevice, newByDevice, m_sizeX, m_sizeY );
	TextureDevice( textureDevice, irbImageDevice, deviceMedianImg, m_sizeX, m_sizeY );

	cudaFree(newRgDevice);
	cudaFree(newByDevice);
	cudaFree(irbImageDevice);
	cudaFree(hueDevice);
	cudaFree(saturationDevice);
	cudaFree(textureDevice);

	tmpScale = CalcScale(orgScale, 12);
	cudaFree(ngbDevice);
	cudaMalloc(&ngbDevice, 2 * tmpScale * tmpScale * sizeof(int) );
	cudaMemcpy(ngbDevice, m_ngb, 2 * tmpScale * tmpScale * sizeof(int), cudaMemcpyHostToDevice);
	
	MedianFilterDevice( textureDevice, deviceMedianImg, tmpScale, ngbDevice, m_sizeX, m_sizeY);

	unsigned char *mapDevice;
	cudaMalloc( &mapDevice, m_stride * sizeof(unsigned char) );
	GenerateMapDevice ( deviceMedianImg, hueDevice, saturationDevice, mapDevice, m_sizeX, m_sizeY );
	cudaMemcpy(map, mapDevice, m_stride * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	cudaFree(deviceMedianImg);
	cudaFree(hueDevice);
	cudaFree(saturationDevice);
	cudaFree(ngbDevice);
}

int SkinFilter::CalcScale(const int initScale, const int factor)
{
	int scale;
	scale = initScale * factor;
	if( !(scale % 2) )
		scale++;
	GenNgb( scale );

	printf( "Calculated scale is %d\n", scale );

	return scale;
}

void SkinFilter::GenNgb( const int scale )
{
	if( m_ngb != NULL )
		delete m_ngb;
	
	const int factor  = ( scale - 1 ) / 2;
	int currPos = 0;

	m_ngb = new int[2 * scale * scale ];

	for( int i = -factor; i <= factor; i++ )
	{
		for( int j = -factor; j <= factor; j++ )
		{
			m_ngb[ currPos   ] = i;
			m_ngb[ currPos+1 ] = j;

			currPos += 2;
		}
	}
}


