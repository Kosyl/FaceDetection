
#include "DCIDetect.h"

#include "opencv/cv.h"
#include "opencv/highgui.h"

DCIDetect::DCIDetect( void )
{

}
	
DCIDetect::~DCIDetect( void )
{

}

std::vector<HaarRectangle> DCIDetect::Run(unsigned char *& img, unsigned char *& imgGrey, const int sizeX, const int sizeY)
{
	unsigned char* map;
	map = new unsigned char[sizeX * sizeY];

	m_skinFilter.Filter( img, map, sizeX, sizeY);

	cv::Mat imgOut( sizeY, sizeX , CV_8UC1 );
	int eltNo = 0;
	
	for( int i = 0; i < sizeY; i++ )
	{
		for( int j = 0; j < sizeX; j++ )
		{
			imgOut.at<unsigned char>(i,j) = map[ eltNo ];
			eltNo++;
		}
	}

	cv::imwrite( "result.jpg", imgOut );

	std::vector<HaarRectangle> result = m_faceFilter.Filter(imgGrey, map, sizeX, sizeY);
	eltNo = 0;
	
	for( int i = 0; i < sizeY; i++ )
	{
		for( int j = 0; j < sizeX; j++ )
		{
			imgOut.at<unsigned char>(i,j) = map[ eltNo ];
			eltNo++;
		}
	}

	cv::imwrite( "result1.jpg", imgOut );
	
	delete[] map;
	return result;
}