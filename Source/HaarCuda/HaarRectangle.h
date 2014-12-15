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
		weight(weight),
		sizeScale(1.0f),
		scaledWeight(1.0f),
		valid(true)
	{
		right = left + width;
		bottom = top + height;
	}

	HaarRectangle() :
		top(0),
		left(0),
		width(0),
		height(0),
		weight(0.0f),
		sizeScale(1.0f),
		scaledWeight(1.0f),
		right(0),
		bottom(0),
		valid(false)
	{

	}

	HaarRectangle(const HaarRectangle& rhs) :
		top(rhs.top),
		left(rhs.left),
		width(rhs.width),
		height(rhs.height),
		weight(rhs.weight),
		sizeScale(rhs.sizeScale),
		scaledWeight(rhs.scaledWeight),
		right(rhs.right),
		bottom(rhs.bottom),
		valid(rhs.valid)
	{

	}

	HaarRectangle& HaarRectangle::operator=(HaarRectangle rhs)
	{
		top = (rhs.top);
		left = (rhs.left);
		width = (rhs.width);
		height = (rhs.height);
		weight = (rhs.weight);
		sizeScale = (rhs.sizeScale);
		scaledWeight = rhs.scaledWeight;
		right = rhs.right;
		bottom = rhs.bottom;
		valid = rhs.valid;
		return *this;
	}

	~HaarRectangle()
	{

	}

	UInt top, left, width, height, bottom, right;

	float weight, sizeScale, scaledWeight;

	bool valid;

	void scaleSize(double scale)
	{
		sizeScale = scale;
	}
	UInt scaledX()
	{
		return (UInt)(this->left * sizeScale);
	}
	UInt scaledY()
	{
		return (UInt)(this->top * sizeScale);
	}
	UInt scaledWidth()
	{
		return (UInt)(this->width * sizeScale);
	}
	UInt scaledHeight()
	{
		return (UInt)(this->height * sizeScale);
	}
	UInt scaledArea()
	{
		return (UInt)(scaledWidth() * scaledHeight());
	}
};

#endif