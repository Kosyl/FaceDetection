#include <cstdio>
#include <stdio.h>
#include "HaarAlgorithm.h"
#include "IntegralImage.h"
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
	//const char* pathIn = "zdj.jpg";
	//const char* pathOutGrey = "obrazTestOutGrey.jpg";
	//const char* pathOutColor = "obrazTestOutColor.jpg";

	//unsigned char *imgColor = NULL;
	//unsigned char *imgGrey = NULL;
	//unsigned char *imgRed = NULL;
	//unsigned char *transposed = NULL;

	//ImgIO imgIO;

	//imgIO.ReadImgColor(pathIn, imgColor);

	//imgIO.ColorToRed(imgColor, imgRed);
	//imgIO.ColorToGray(imgColor, imgGrey);
	////imgIO.Transpose(imgGrey, transposed);

	//IntegralImage image(imgIO.getSizeX(), imgIO.getSizeY(), imgGrey);
	//HaarAlgorithm alg;
	//std::vector<HaarRectangle> result = alg.execute(&image);

	//for (std::vector<HaarRectangle>::iterator i = result.begin(); i != result.end(); ++i)
	//{
	//	mark(*i, imgGrey, imgIO.getSizeX());
	//}

	//imgIO.WriteImgColor(pathOutColor, imgColor);
	//imgIO.WriteImgGrey(pathOutGrey, imgGrey);

	//delete[] imgColor;
	//delete[] imgGrey;
	//delete[] imgRed;
	//delete[] transposed;

	//return 0;
	
	const char* pathIn = "lena_color.jpg";
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

	return 0;
}