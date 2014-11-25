#include "stdafx.h"
#include "FaceFilter.h"
#include "assert.h"

typedef unsigned char uchar;

FaceFilter::FaceFilter( void )
{

}

FaceFilter::~FaceFilter( void )
{

}

void FaceFilter::Filter( unsigned char *&img, unsigned char *&map, const int sizeX, const int sizeY )
{
	assert( sizeX );
	assert( sizeY );
	
	uchar *closed;
	uchar *stretched;
	uchar *holes;


	closed = new uchar[ sizeX * sizeY ];
	
	Close( map, closed, sizeX, sizeY );
	Mask( img, closed, sizeX, sizeY );
	
	StretchColor( img, sizeX, sizeY );
	Negate( img, img, sizeX, sizeY );

	delete[] closed;
}

void FaceFilter::Mask( unsigned char *&img, unsigned char *&mask, const int sizeX, const int sizeY )
{
	assert( img );
	
	for( int i = 0; i < sizeX * sizeY; i++ )
	{
		if( mask[i] == 0 )
			img[i] = 0;
	}
}

void FaceFilter::Close( unsigned char *&img, unsigned char *&out, const int sizeX, const int sizeY )
{
	const int eltSize = 7;

	int   *structElt;
	uchar *tempImg;
	structElt = new int[ 2 * eltSize * eltSize ];
	tempImg   = new uchar[ sizeX * sizeY ];

	CreateStructElt( structElt, eltSize );

	//dilate
	for( int i = 0; i < sizeX; i++)
	{	
		for( int j = 0; j < sizeY; j++ )
		{
			if( i < eltSize || i > sizeX - eltSize || j < eltSize || j > sizeY - eltSize )
			{
				tempImg[ i + sizeX * j ] = 0;
			}
			else
			{
				int val = 0;

				for( int k = 0; k < 2 * eltSize * eltSize; k += 2 )
				{
					val += img[ i + structElt[k] + (sizeX * (j + structElt[k+1])) ];
				}
			
				if( val )
					val = 255;

				for( int k = 0; k < 2 * eltSize * eltSize; k += 2 )
				{
					tempImg[ i + structElt[k] + (sizeX * (j + structElt[k+1])) ] = val;
				}
			}
		}
	}

	//erode
	for( int i = 0; i < sizeX; i++)
	{
		for( int j = 0; j < sizeY; j++ )
		{
			uchar val = 0;

			if( i < eltSize || i > sizeX - eltSize || j < eltSize || j > sizeY - eltSize )
			{
				out[ i + sizeX * j ] = 0;
			}
			else
			{
				for( int k = 0; k < 2 * eltSize * eltSize; k += 2 )
				{
					if( tempImg[ i + structElt[k] + (sizeX * (j + structElt[k+1])) ] == 0 )
					{
						val = 0;
						break;
					}

					val = 255;
				}
			
				for( int k = 0; k < 2 * eltSize * eltSize; k += 2 )
				{
					out[ i + structElt[k] + (sizeX * (j + structElt[k+1])) ] = val;
				}
			}
		}
	}

	delete[] structElt;
	delete[] tempImg;
}

void FaceFilter::CreateStructElt( int *&elt, const int eltSize )
{
	const int factor = ( eltSize - 1 ) / 2;
	int currPos = 0;

	printf( "Generating ngb: \n" );

	for( int i = -factor; i <= factor; i++ )
	{
		for( int j = -factor; j <= factor; j++ )
		{
			elt[ currPos   ] = i;
			elt[ currPos+1 ] = j;

			currPos += 2;
		}
	}
}

void FaceFilter::StretchColor( unsigned char *&img, const int sizeX, const int sizeY )
{
	uchar maxVal = 0;
	// uchar minVal = 255;

	for( int i = 0; i < sizeX * sizeY; i++ )
	{
		const int pxlVal = (int)img[i];

		/*if( pxlVal < minVal )
			minVal = pxlVal;*/
		
		if( pxlVal > maxVal )
			maxVal = pxlVal;
	}

	for( int i = 0; i < sizeX * sizeY; i++ )
	{
		const int newVal = (((float)img[i] / (float)maxVal) * 255.f );

		if( newVal > 95 && newVal < 240 )
			img[i] = newVal;
		else
			img[i] = 0;
	}

}

void FaceFilter::Negate( unsigned char *&img, unsigned char *&out, const int sizeX, const int sizeY )
{
	for( int i = 0; i < sizeX * sizeY; i++ )
	{
		if( !img[i] )
			img[i] = 255;
		else 
			img[i] = 0;
	}

	uchar* holes = new uchar[ sizeX * sizeY ];

	for( int i = 0; i < sizeX * sizeY; i++ )
		holes[i] = img[i];

	int i = 0;
	int j = 0;
	
	FindHoles( img, holes, 0, 0, sizeX, sizeY );
	
	for( int i = 0; i < sizeX * sizeY; i++ )
		img[i] = holes[i];

	delete[] holes;
}

void FaceFilter::FindHoles( unsigned char *&img, unsigned char *&holes, int x, int y, const int sizeX, const int sizeY )
{
	if( x < 0 || x >= sizeX || y < 0 || y >= sizeY )
		return;
	
	holes[ x + sizeX * y ] = 0;
	
	for( int i = -1; i <= 1; i++ )
	{
		for( int j = -1; j <= 1; j++ )
		{
			if(i == 0 && j == 0 )
				continue;

			if( img[ (x + i) + sizeX * (y + j) ] == 255 && holes[ (x + i) + sizeX * (y + j) ] == 255 )
			{
				FindHoles( img, holes, x+i, y+j, sizeX, sizeY );
			}
		}
	}
}