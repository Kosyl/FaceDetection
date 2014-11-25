#include "stdafx.h"
#include <cstdio>
#include <stdio.h>

#include "imgIO.hpp"
#include "DCIDetect.h"

int main( )
{
	const char* pathIn       = "1m.jpg";
	const char* pathOutGrey  = "obrazTestOutGrey.jpg";
	const char* pathOutColor = "obrazTestOutColor.jpg";

	unsigned char *imgColor = NULL;
	unsigned char *imgGrey  = NULL;
	
	ImgIO imgIO;
	imgIO.ReadImgColor ( pathIn, imgColor  );

	DCIDetect detect;

	const int imgSizeX = imgIO.GetXSize();
	const int imgSizeY = imgIO.GetYSize();

	imgIO.ColorToGray  ( imgColor, imgGrey );

	detect.Run( imgColor, imgGrey, imgSizeX, imgSizeY );
	imgIO.WriteImgGrey( pathOutGrey, imgGrey );
	
	delete[] imgColor;

	return 0;
}