#pragma once

#include <cstdlib>

#include "FaceFilter.h"
#include "SkinFilter.h"
#include <vector>
#include "HaarRectangle.h"

class DCIDetect
{
public: 
	 DCIDetect( void );
	~DCIDetect( void );

	std::vector<HaarRectangle> Run(unsigned char *& img, unsigned char *& imgGrey, const int sizeX, const int sizeY);

private:
	
	SkinFilter m_skinFilter;
	FaceFilter m_faceFilter;
};