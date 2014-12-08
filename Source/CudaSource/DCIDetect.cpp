
#include "DCIDetect.h"

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

	std::vector<HaarRectangle> result = m_faceFilter.Filter(imgGrey, map, sizeX, sizeY);

	delete[] map;

	return result;
}