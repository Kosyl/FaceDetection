#include "imgIO.hpp"
#include <cmath>

ImgIO::ImgIO( void )
{

}

ImgIO::~ImgIO( void )
{
	if( m_vidInp.isOpened() )
		m_vidInp.release();

	if( m_vidOut.isOpened() )
		m_vidOut.release();
}

void ImgIO::ReadImgColor( const char* path, unsigned char *&img )
{
	const int channels = 3;
	
	cv::Mat tempImg = cv::imread( path, 1 );

	m_sizeX = tempImg.cols;
	m_sizeY = tempImg.rows;

	printf( "Img sizeX = %d, sizeY = %d\n", m_sizeX, m_sizeY );

	if( img != NULL )
		delete[] img;

	img = new unsigned char[ m_sizeX * m_sizeY * channels];
	Mat2Char( img, tempImg );
}

void ImgIO::DeviceColorToGray( unsigned char *&imgColor, unsigned char *&imgGray )
{
	// ToGrey( imgColor, imgGray );
}

void ImgIO::ColorToGray( unsigned char *&imgColor, unsigned char *&imgGray )
{
	if( imgGray == NULL )
		imgGray = new unsigned char[ m_sizeX * m_sizeY ];

	const double redWeight   = 0.21f;
	const double greenWeight = 0.72f;
	const double blueWeight  = 0.07f;

	int eltNo = 0;
	int colorNo = 0;

	for( int i = 0; i < m_sizeY; i++ )
	{
		for( int j = 0; j < m_sizeX; j++ )
		{	
			imgGray[ eltNo ] = redWeight   * imgColor[ colorNo ] + greenWeight * imgColor[ colorNo + 1 ] + blueWeight  * imgColor[ colorNo + 2 ];

			eltNo++;
			colorNo += 3;
		}
	}
}

void ImgIO::ColorToRed(unsigned char *&imgColor, unsigned char *&imgRed)
{
	if (imgRed == NULL)
		imgRed = new unsigned char[m_sizeX * m_sizeY];

	const double redWeight = 0.21f;
	const double greenWeight = 0.72f;
	const double blueWeight = 0.07f;

	int eltNo = 0;
	int colorNo = 0;

	for (int i = 0; i < m_sizeY; i++)
	{
		for (int j = 0; j < m_sizeX; j++)
		{
			imgRed[eltNo] = imgColor[colorNo];

			eltNo++;
			colorNo += 3;
		}
	}
}

void ImgIO::WriteImgGrey( const char* path, unsigned char *&img )
{
	cv::Mat tempImg( m_sizeY, m_sizeX , CV_8UC1 );
	Char2MatGrey( img, tempImg );

	int ret = cv::imwrite( path, tempImg );

	if( !ret )
		printf( "Cannot save img %s\n", path );
}

void ImgIO::WriteImgColor( const char* path, unsigned char *&img )
{
	cv::Mat tempImg( m_sizeY, m_sizeX , CV_8UC3 );
	Char2MatColor( img, tempImg );

	int ret = cv::imwrite( path, tempImg );

	if( !ret )
		printf( "Cannot save img %s\n", path );
}

void ImgIO::OpenInputStream ( const char* path )
{
	m_vidInp.open( path );

	if( !m_vidInp.isOpened() )
	{
		printf( "Cannot Open VidInp: '%s'!\n", path ); 
		exit(1); 
	}
}

void ImgIO::OpenOutputStream( const char* path )
{
	m_vidInp.open( path );

	if( !m_vidInp.isOpened() )
	{
		printf( "Cannot Open VidInp: '%s'!\n", path ); 
		exit(1); 
	}
}

void ImgIO::ReadFromStream( unsigned char* &img )
{
	cv::Mat imgCV;

	if( !m_vidInp.grab( ) )
	{
		printf( "Cannot grab image\n" );
		return;
	}

	if( !m_vidInp.retrieve( imgCV ) )
	{
		printf( "Cannot grab image\n" );
		return;
	}

	Mat2Char( img, imgCV );
}

void ImgIO::WriteToStream( unsigned char* &img )
{
	cv::Mat imgCV;
	Char2MatColor( img, imgCV );
	m_vidOut.write( imgCV );
}

void ImgIO::Mat2Char( unsigned char* &chImg, const cv::Mat &matImg )
{
	assert( chImg );

	int eltNo = 0;
	cv::Vec3b pVal(0,0,0);

	for( int i = 0; i < m_sizeY; i++ )
	{
		for( int j = 0; j < m_sizeX; j++ )
		{
			pVal = matImg.at<cv::Vec3b>(i,j);
			
			chImg[ eltNo + 2 ] = pVal[0];
			chImg[ eltNo + 1 ] = pVal[1];
			chImg[ eltNo     ] = pVal[2];
			eltNo += 3;
		}
	}
}

void ImgIO::Char2MatGrey( unsigned char* &chImg, cv::Mat &matImg )
{
	int eltNo = 0;
	
	for( int i = 0; i < m_sizeY; i++ )
	{
		for( int j = 0; j < m_sizeX; j++ )
		{
			matImg.at<uchar>(i,j) = chImg[ eltNo ];

			eltNo++;
		}
	}
}

void ImgIO::Char2MatColor( unsigned char* &chImg, cv::Mat &matImg )
{
	int eltNo = 0;
	cv::Vec3b pVal(0,0,0);
	
	for( int i = 0; i < m_sizeY; i++ )
	{
		for( int j = 0; j < m_sizeX; j++ )
		{
			pVal[0] = chImg[ eltNo + 2 ];
			pVal[1] = chImg[ eltNo + 1 ];
			pVal[2] = chImg[ eltNo + 0 ];

			matImg.at<cv::Vec3b>(i,j) = pVal;

			eltNo += 3;
		}
	}
}

void ImgIO::Transpose(unsigned char *&source, unsigned char *&dst)
{
	if (dst == NULL)
		dst = new unsigned char[m_sizeX * m_sizeY];

	int eltNo = 0;

	for (int i = 0; i < m_sizeX; i++)
	{
		for (int j = 0; j < m_sizeY; j++)
		{
			dst[eltNo] = source[j*m_sizeX + i];

			eltNo++;
		}
	}
}
