#pragma once

#include <cstdlib>

#include "FaceFilter.h"
#include "SkinFilter.h"

class DCIDetect
{
public: 
	 DCIDetect( void );
	~DCIDetect( void );

	void Run( unsigned char *& img, unsigned char *& imgGrey, const int sizeX, const int sizeY );

private:
	
	SkinFilter m_skinFilter;
	FaceFilter m_faceFilter;
};