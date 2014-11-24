#include <cstdio>
#include <stdio.h>
#include "HaarAlgorithm.h"
#include "IntegralImage.h"
#include "imgIO.hpp"
#include <vector>

void mark(HaarRectangle& rect, unsigned char* imgGrey, UInt stride)
{
	for (int i = 0; i < rect.height; ++i)
	{
		imgGrey[(rect.top*stride) + rect.left + i] = 250; // gora
		imgGrey[((rect.top + rect.height)*stride) + rect.left + i] = 250; //dol
		imgGrey[(rect.top + i)*stride + rect.left] = 250;
		imgGrey[(rect.top + i)*stride + rect.left + rect.width] = 250;
	}
}

int main()
{
	const char* pathIn = "new.jpg";
	const char* pathOutGrey = "obrazTestOutGrey.jpg";
	const char* pathOutColor = "obrazTestOutColor.jpg";

	unsigned char *imgColor = NULL;
	unsigned char *imgGrey = NULL;
	unsigned char *imgRed = NULL;
	unsigned char *transposed = NULL;

	ImgIO imgIO;

	imgIO.ReadImgColor(pathIn, imgColor);

	imgIO.ColorToRed(imgColor, imgRed);
	imgIO.ColorToGray(imgColor, imgGrey);
	//imgIO.Transpose(imgGrey, transposed);

	IntegralImage image(imgIO.getSizeX(), imgIO.getSizeY(), imgGrey);
	HaarAlgorithm alg;
	std::vector<HaarRectangle> result = alg.execute(&image);

	for (std::vector<HaarRectangle>::iterator i = result.begin(); i != result.end(); ++i)
	{
		mark(*i, imgGrey, imgIO.getSizeX());
	}

	imgIO.WriteImgColor(pathOutColor, imgColor);
	imgIO.WriteImgGrey(pathOutGrey, imgGrey);

	delete[] imgColor;
	delete[] imgGrey;
	delete[] imgRed;
	delete[] transposed;

	return 0;
}