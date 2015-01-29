#include "IntegralImage.h"
#include <assert.h>
#include <iostream>
#include <algorithm>


IntegralImage::IntegralImage(UInt imWidth, UInt imHeight, unsigned char *picture, float inScale)
{
	scale = inScale;
	width = static_cast<UInt>(scale > 1.0f ? imWidth / scale : imWidth);
	height = static_cast<UInt>(scale > 1.0f ? imHeight / scale : imHeight);

	int lastWindowX = width - BASE_CLASSIFIER_SIZE, lastWindowY = height - BASE_CLASSIFIER_SIZE;
	assert(lastWindowY > 0 && lastWindowX > 0);
	int windowStep = BASE_CLASSIFIER_SIZE >> WINDOW_STEP_SHIFT;
	int pixelsPerTile = windowStep * TILE_SIZE;

	numTilesX = (width) / pixelsPerTile + 1;
	numTilesY = (height) / pixelsPerTile + 1;
	totalWidth = numTilesX*pixelsPerTile + BASE_CLASSIFIER_SIZE + 2;
	totalHeight = numTilesY*pixelsPerTile + BASE_CLASSIFIER_SIZE + 2;
	stride = totalWidth;

	/*this->originalX = new UInt[width*height];
	this->originalY = new UInt[width*height];
	memset(originalX, 0, width*height*sizeof(UInt));
	memset(originalY, 0, width*height*sizeof(UInt));*/

	unsigned char* scaledImg = new unsigned char[width*height];

	for (size_t x = 0; x < width; x++)
	{
		for (size_t y = 0; y < height; y++)
		{
			int scaledX = static_cast<UInt>(x*scale < imWidth ? x*scale : imWidth);
			int scaledY = static_cast<UInt>(y*scale < imHeight ? y*scale : imHeight);
			scaledImg[y*width + x] = picture[scaledY*imWidth + scaledX];
		}
	}

	this->values = new UInt[totalWidth*totalHeight];
	this->values2 = new UInt[totalWidth*totalHeight];
	memset(values, 0, totalHeight*totalWidth*sizeof(UInt));
	memset(values2, 0, totalHeight*totalWidth*sizeof(UInt));

	calcFromPicture(scaledImg);

	calcWeights();

	delete[] scaledImg;
}

void IntegralImage::calcWeights()
{
	//todo: mozna sprobowac zrownoleglic
	weightsStride = totalWeightsWidth = numTilesX*TILE_SIZE;
	totalWeightsHeight = numTilesY*TILE_SIZE;

	this->weights = new float[totalWeightsWidth*totalWeightsHeight];
	memset(weights, 0, totalWeightsWidth*totalWeightsHeight*sizeof(float));

	for (size_t x = 0; x < totalWeightsWidth; x++)
	{
		for (size_t y = 0; y < totalWeightsHeight; y++)
		{
			float mean = getSumInRect(x, y, BASE_CLASSIFIER_SIZE, BASE_CLASSIFIER_SIZE) * INV_AREA;
			float factor = getSum2InRect(x, y, BASE_CLASSIFIER_SIZE, BASE_CLASSIFIER_SIZE) * INV_AREA - (mean * mean);

			weights[y*weightsStride+x] = (factor >= 0) ? std::sqrt(factor) : 1;
		}
	}
}

IntegralImage::~IntegralImage()
{
	delete[] values;
	delete[] values2;
}

void IntegralImage::calcFromPicture(unsigned char* picture)
{
	unsigned char* src = picture;

	values[0] = 0;
	values2[0] = 0;

	for (size_t i = 1; i <= width; ++i)
	{
		values[i] = 0;
		values2[i] = 0;
	}
	for (size_t i = 1; i <= height; ++i)
	{
		values[i*stride] = 0;
		values2[i*stride] = 0;
	}

	for (size_t y = 1; y <= height; ++y)
	{
		int yCurrent = stride * (y);
		int yBefore = stride * (y - 1);

		for (size_t x = 1; x <= width; ++x)
		{
			int p1 = *src;
			int p2 = p1 * p1;

			int idx = yCurrent + x;
			int bottomLeft = yCurrent + (x - 1);
			int upperRight = yBefore + (x);
			int upperLeft = yBefore + (x - 1);

			values[idx] = p1 + values[bottomLeft] + values[upperRight] - values[upperLeft];
			values2[idx] = p2 + values2[bottomLeft] + values2[upperRight] - values2[upperLeft];

			++src;
		}
	}
}

UInt IntegralImage::getSumInRect(UInt left, UInt top, UInt windowWidth, UInt windowHeight)
{
	int a = stride * (top)+(left);
	int b = stride * (top + windowHeight) + (left + windowWidth);
	int c = stride * (top + windowHeight) + (left);
	int d = stride * (top)+(left + windowWidth);

	return values[a] + values[b] - values[c] - values[d];
}

UInt IntegralImage::getSum2InRect(UInt left, UInt top, UInt windowWidth, UInt windowHeight)
{
	int a = stride * (top)+(left);
	int b = stride * (top + windowHeight) + (left + windowWidth);
	int c = stride * (top + windowHeight) + (left);
	int d = stride * (top)+(left + windowWidth);

	return values2[a] + values2[b] - values2[c] - values2[d];
}