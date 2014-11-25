
#include "SkinFilter.h"
#include <cmath>

#include "opencv/cv.h"
#include "opencv/highgui.h"

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

	float *irbImage;
	irbImage = new float[ m_stride * 3 ]; // I[], Rg[], By[]

	RGB2IRB( img, irbImage );

	tmpScale = orgScale * 8;
	if( !(tmpScale % 2) )
		tmpScale++;
	GenNgb( tmpScale );

	
	float *medianImg = new float[m_stride];
	MedianFilter( irbImage, medianImg, tmpScale );

	float *newRg = new float[m_stride];
	float *newBy = new float[m_stride];

	float *Rg = irbImage + m_stride;
	float *By = irbImage + 2 * m_stride;

	float *hue        = new float[m_stride];
	float *saturation = new float[m_stride];
	float *texture    = new float[m_stride];

	tmpScale = 4 * orgScale;
	if( !(tmpScale % 2) )
		tmpScale++;
	GenNgb( tmpScale );

	MedianFilter( Rg, newRg, tmpScale );
	MedianFilter( By, newBy, tmpScale );
	HueSaturation( hue, saturation, newRg, newBy );
	Texture( texture, irbImage, medianImg );

	tmpScale = 12 * orgScale;
	if( !(tmpScale % 2) )
		tmpScale++;
	GenNgb( tmpScale );

	MedianFilter( texture, medianImg, tmpScale    );
	GenerateMap ( medianImg, hue, saturation, map );
	WriteMap( map );

	delete[] irbImage;
	delete[] newRg;
	delete[] newBy;
	delete[] hue;
	delete[] saturation;
	delete[] texture;
	delete[] medianImg;
}

void SkinFilter::GenerateMap( float *& medianImg, float *& hue, float *& saturation, unsigned char *& map )
{
	for( int i = 0; i < m_stride; i++ )
	{
		bool m = medianImg[i] < 4.5;
		bool h = hue[i] > 120 && hue[i] < 160;
		bool s = saturation[i] > 10 && saturation[i] < 60;

		if( m && h && s )
		{
			map[i] = 255;
			continue;
		}

		m = medianImg[i] < 4.5;
		h = hue[i] > 150 && hue[i] < 180;
		s = saturation[i] > 20 && saturation[i] < 80;

		if( m && h && s )
		{
			map[i] = 255;
			continue;
		}

		map[i] = 0;
	}
}

void SkinFilter::WriteMap( unsigned char *&map )
{
	cv::Mat img( m_sizeY, m_sizeX , CV_8UC1 );

	int eltNo = 0;
	
	for( int i = 0; i < m_sizeY; i++ )
	{
		for( int j = 0; j < m_sizeX; j++ )
		{
			img.at<uchar>(i,j) = map[ eltNo ];
			eltNo++;
		}
	}

	cv::imwrite( "result.jpg", img );
}


void SkinFilter::RGB2IRB( unsigned char *&imgIn, float *&imgOut )
{
	const int stride = m_sizeX * m_sizeY;
	
	for( int i = 0; i < m_sizeX * m_sizeY; i++ )
	{
		float R = float( imgIn[3 * i    ] );
		float G = float( imgIn[3 * i + 1] );
		float B = float( imgIn[3 * i + 2] );

		imgOut[              i ] = ( L(R) + L(B) + L(G) ) / 3;
		imgOut[ stride     + i ] = L(R) - L(G);
		imgOut[ 2 * stride + i ] = L(B) - ( L(G) + L(R) ) / 2;
	}
}

void SkinFilter::MedianFilter( float * & img, float * & medianImg, int scale )
{
	float *window = new float[ scale * scale ];
	
	for( int i = 0; i < m_sizeX; i++)
	{
		for( int j = 0; j < m_sizeY; j++ )
		{
			if( i < scale || i > m_sizeX - scale || j < scale || j > m_sizeY - scale )
			{
				medianImg[ i + m_sizeX * j ] = img[ i + m_sizeX * j ];
			}
			else
			{
				int windCnt = 0;

				for( int k = 0; k < 2 * scale * scale; k += 2 )
				{
					const float val = img[ i + m_ngb[k] + (m_sizeX * (j + m_ngb[k+1])) ];
					window[windCnt] = val;
					windCnt++;
				}
			
				float median = GetMedian( window, scale );
			
				medianImg[ i + m_sizeX * j ] = median;
			}
		}
	}

	delete[] window;
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

float SkinFilter::GetMedian( float *& img, int scale )
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

void SkinFilter::Texture( float *&texture, float *&img, float *&imgFiltered )
{
	for( int i=0; i < m_sizeX * m_sizeY; i++ )
	{
		texture[i] = abs( img[i] - imgFiltered[i] );
	}
}

void SkinFilter::HueSaturation( float *&hue, float *&saturation, float *& By, float *&Rb )
{
	for( int i=0; i < m_sizeX * m_sizeY; i++ )
	{
		const float  Rg_ = By[i];
		const float  By_ = Rb[i];

		hue[i]        = ( atan2f( Rg_, By_ ) / 3.14 ) * 180;
		saturation[i] =   hypotf( Rg_, By_ );
	}
}

