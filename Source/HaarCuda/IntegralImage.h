#ifndef _INTEGRAL_IMAGE_H_
#define _INTEGRAL_IMAGE_H_

#include "TypeDef.h"

class IntegralImage
{
private:

	void calcFromPicture(unsigned char* picture);
	void calcWeights();

public:

	IntegralImage(UInt width, UInt height, unsigned char *picture, float scale);
	~IntegralImage();

	//wazne piksele
	UInt width;
	UInt height;
	//z dopelnieniem do pelnych kafli
	UInt totalWidth;
	UInt totalHeight;
	UInt totalWeightsWidth;
	UInt totalWeightsHeight;

	//macierze totalWidth*totalHeight
	UInt *values;
	UInt *values2;
	float* weights;
	UInt *originalX;
	UInt *originalY;

	//wiersz probek
	UInt stride;
	//wiersz wag
	UInt weightsStride;

	UInt numTilesX;
	UInt numTilesY;

	float scale;

	UInt getSumInRect(UInt top, UInt left, UInt width, UInt height);
	UInt getSum2InRect(UInt top, UInt left, UInt width, UInt height);
};

#endif