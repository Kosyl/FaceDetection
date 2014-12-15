#include <cstdio>
#include <stdio.h>
#include "imgIO.hpp"
#include "DCIDetect.h"
#include <vector>

void mark(HaarRectangle& rect, unsigned char* imgGrey, UInt stride)
{
	for (int i = 0; i < rect.height; ++i)
	{
		imgGrey[(rect.top + i)*stride + rect.left] = 250;
		imgGrey[(rect.top + i)*stride + rect.left + rect.width] = 250;
	}
	for (int i = 0; i < rect.width; ++i)
	{
		imgGrey[(rect.top*stride) + rect.left + i] = 250; // gora
		imgGrey[((rect.top + rect.height)*stride) + rect.left + i] = 250; //dol
	}
}

int main()
{
	const char* pathIn = "3m.jpg";
	const char* pathOutGrey = "obrazTestOutGrey.jpg";
	const char* pathOutColor = "obrazTestOutColor.jpg";

	unsigned char *imgColor = NULL;
	unsigned char *imgGrey = NULL;
	unsigned char *imgClean = NULL;

	ImgIO imgIO;
	imgIO.ReadImgColor(pathIn, imgColor);

	DCIDetect detect;

	const int imgSizeX = imgIO.getSizeX();
	const int imgSizeY = imgIO.getSizeY();

	imgIO.ColorToGray(imgColor, imgGrey);
	imgIO.ColorToGray(imgColor, imgClean);

	std::vector<HaarRectangle> result = detect.Run(imgColor, imgGrey, imgSizeX, imgSizeY);

	for (std::vector<HaarRectangle>::iterator i = result.begin(); i != result.end(); ++i)
	{
		mark(*i, imgClean, imgIO.getSizeX());
	}
	 
	imgIO.WriteImgGrey(pathOutGrey, imgClean);

	delete[] imgColor;
	getchar();

	return 0;
}