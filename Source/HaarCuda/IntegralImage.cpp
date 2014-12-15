#include "IntegralImage.h"


IntegralImage::IntegralImage(UInt width, UInt height, unsigned char *picture) :
width(width),
height(height),
stride(width + 1)
{
	this->values = new UInt[(width+1)*(height+1)];
	this->values2 = new UInt[(width+1)*(height+1)];
	calcFromPicture(picture);
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

	for (int i = 1; i <= width; ++i)
	{
		values[i] = 0;
		values2[i] = 0;
	}
	for (int i = 1; i <= height; ++i)
	{
		values[i*stride] = 0;
		values2[i*stride] = 0;
	}

	for (int y = 1; y <= height; ++y)
	{
		int yCurrent = stride * (y);
		int yBefore = stride * (y - 1);

		for (int x = 1; x <= width; ++x)
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