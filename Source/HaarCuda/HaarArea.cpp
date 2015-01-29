#include "HaarArea.h"


void HaarArea::setWeight( double weight)
{
	if (this->numRectangles == 2)
	{
		rectangles[1].weight = static_cast<float>(rectangles[1].weight * (weight));

		rectangles[0].weight = (-(rectangles[1].area * rectangles[1].weight) / rectangles[0].area);
	}
	else // this.numRectangles == 3
	{
		rectangles[2].weight = static_cast<float>(rectangles[2].weight * (weight));

		rectangles[1].weight = static_cast<float>(rectangles[1].weight * (weight));

		rectangles[0].weight = static_cast<float>((-(rectangles[1].area * rectangles[1].weight + rectangles[2].area * rectangles[2].weight) / (rectangles[0].area)));
	}
}

double HaarArea::checkMatch(IntegralImage* image, UInt top, UInt left, double scaleFactor)
{
	double sum = 0.0;

	for (size_t i = 0; i < this->numRectangles; ++i)
	{
		HaarRectangle& rect = this->rectangles[i];
		sum += image->getSumInRect(left + rect.left, top + rect.top, rect.width, rect.height) * rect.weight;
	}

	return sum;
}

HaarArea::~HaarArea()
{
}

HaarArea::HaarArea(double threshold, double valueIfSmaller, double valueIfBigger, HaarRectangle rectangle1, HaarRectangle rectangle2):
threshold(static_cast<float>(threshold)),
valueIfSmaller(static_cast<float>(valueIfSmaller)),
valueIfBigger(static_cast<float>(valueIfBigger)),
numRectangles(2)
{
	rectangles[0] = rectangle1;
	rectangles[1] = rectangle2;
}

HaarArea::HaarArea(double threshold, double valueIfSmaller, double valueIfBigger, HaarRectangle rectangle1, HaarRectangle rectangle2, HaarRectangle rectangle3) :
threshold(static_cast<float>(threshold)),
valueIfSmaller(static_cast<float>(valueIfSmaller)),
valueIfBigger(static_cast<float>(valueIfBigger)),
numRectangles(3)
{
	rectangles[0] = rectangle1;
	rectangles[1] = rectangle2;
	rectangles[2] = rectangle3;
}