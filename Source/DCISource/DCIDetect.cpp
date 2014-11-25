#include "stdafx.h"

#include "DCIDetect.h"

DCIDetect::DCIDetect( void )
{

}
	
DCIDetect::~DCIDetect( void )
{

}

void DCIDetect::Run( unsigned char *& img, unsigned char *& imgGrey, const int sizeX, const int sizeY )
{
	unsigned char* map;
	map = new unsigned char[sizeX * sizeY];

	m_faceFilter.Filter( img, map, sizeX, sizeY );
	m_skinFilter.Filter( imgGrey, map, sizeX, sizeY );

	delete[] map;
}