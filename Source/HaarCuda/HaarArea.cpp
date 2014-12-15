#include "HaarArea.h"


void HaarArea::setScaleAndWeight(double scale, double weight)
{
	if (this->numRectangles == 2)
	{
		rectangles[1].scaleSize(scale);
		rectangles[1].scaledWeight = rectangles[1].weight * (weight);

		rectangles[0].scaleSize(scale);
		rectangles[0].scaledWeight = (-(rectangles[1].scaledArea() * rectangles[1].scaledWeight) / rectangles[0].scaledArea());
	}
	else // this.numRectangles == 3
	{
		rectangles[2].scaleSize(scale);
		rectangles[2].scaledWeight = rectangles[2].weight * (weight);

		rectangles[1].scaleSize(scale);
		rectangles[1].scaledWeight = rectangles[1].weight * (weight);

		rectangles[0].scaleSize(scale);
		rectangles[0].scaledWeight = (-(rectangles[1].scaledArea() * rectangles[1].scaledWeight + rectangles[2].scaledArea() * rectangles[2].scaledWeight) / (rectangles[0].scaledArea()));
	}
}

double HaarArea::checkMatch(IntegralImage* image, UInt top, UInt left, double scaleFactor)
{
	double sum = 0.0;

	for (int i = 0; i < this->numRectangles; ++i)
	{
		HaarRectangle& rect = this->rectangles[i];
		sum += image->getSumInRect(left + rect.scaledX(), top + rect.scaledY(), rect.scaledWidth(), rect.scaledHeight()) * rect.scaledWeight;
	}

	return sum;
}

HaarArea::~HaarArea()
{
	//delete[] this->rectangles;
}

HaarArea::HaarArea(double threshold, double valueIfSmaller, double valueIfBigger, HaarRectangle rectangle1, HaarRectangle rectangle2):
threshold(threshold),
valueIfSmaller(valueIfSmaller),
valueIfBigger(valueIfBigger),
numRectangles(2)
{
	rectangles[0] = rectangle1;
	rectangles[1] = rectangle2;
}

HaarArea::HaarArea(double threshold, double valueIfSmaller, double valueIfBigger, HaarRectangle rectangle1, HaarRectangle rectangle2, HaarRectangle rectangle3) :
threshold(threshold),
valueIfSmaller(valueIfSmaller),
valueIfBigger(valueIfBigger),
numRectangles(3)
{
	rectangles[0] = rectangle1;
	rectangles[1] = rectangle2;
	rectangles[2] = rectangle3;
}