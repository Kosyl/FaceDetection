#ifndef _HAAR_RECTANGLE_H_
#define _HAAR_RECTANGLE_H_

#include "TypeDef.h"

class HaarRectangle
{
public:
	HaarRectangle(UInt left, UInt top, UInt width, UInt height, double weight) :
		top(top),
		left(left),
		width(width),
		height(height),
		weight(static_cast<float>(weight)),
		valid(true)
	{
		right = left + width;
		bottom = top + height;
		area = width * height;
	}

	HaarRectangle() :
		top(0),
		left(0),
		width(0),
		height(0),
		weight(0.0f),
		right(0),
		bottom(0),
		valid(false),
		area(0)
	{

	}

	HaarRectangle(const HaarRectangle& rhs) :
		top(rhs.top),
		left(rhs.left),
		width(rhs.width),
		height(rhs.height),
		weight(rhs.weight),
		right(rhs.right),
		bottom(rhs.bottom),
		valid(rhs.valid),
		area(rhs.area)
	{

	}

	HaarRectangle& HaarRectangle::operator=(HaarRectangle rhs)
	{
		top = (rhs.top);
		left = (rhs.left);
		width = (rhs.width);
		height = (rhs.height);
		weight = (rhs.weight);
		right = rhs.right;
		bottom = rhs.bottom;
		valid = rhs.valid;
		area = rhs.area;
		return *this;
	}

	~HaarRectangle()
	{

	}

	UInt top, left, width, height, bottom, right, area;

	float weight;

	bool valid;
};

#endif