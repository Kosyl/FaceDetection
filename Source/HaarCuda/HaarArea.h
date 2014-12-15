#ifndef _HAAR_AREA_H_
#define _HAAR_AREA_H_

#include "TypeDef.h"
#include "HaarRectangle.h"
#include "IntegralImage.h"

class HaarArea
{
private:
public:
	float threshold;
	float valueIfBigger;
	float valueIfSmaller;

	UInt numRectangles;
	HaarRectangle rectangles[3];

	HaarArea()
	{

	}

	HaarArea(const HaarArea& rhs) :
		threshold(rhs.threshold),
		valueIfBigger(rhs.valueIfBigger),
		valueIfSmaller(rhs.valueIfSmaller),
		numRectangles(rhs.numRectangles)
	{
		rectangles[0] = rhs.rectangles[0];
		rectangles[1] = rhs.rectangles[1];
		rectangles[2] = rhs.rectangles[2];
	}

	HaarArea(double threshold, double valueIfSmaller, double valueIfBigger, HaarRectangle rectangle1, HaarRectangle rectangle2);
	HaarArea(double threshold, double valueIfSmaller, double valueIfBigger, HaarRectangle rectangle1, HaarRectangle rectangle2, HaarRectangle rectangle3);
	~HaarArea();

	void setScaleAndWeight(double scale, double weight);
	double checkMatch(IntegralImage* image, UInt top, UInt left, double scaleFactor);


	HaarArea& operator=(HaarArea rhs)
	{
		threshold = rhs.threshold;
		valueIfBigger = rhs.valueIfBigger;
		valueIfSmaller = rhs.valueIfSmaller;
		numRectangles = rhs.numRectangles;
		rectangles[0] = rhs.rectangles[0];
		rectangles[1] = rhs.rectangles[1];
		rectangles[2] = rhs.rectangles[2];

		return *this;
	}
};

#endif