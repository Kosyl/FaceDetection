#ifndef _STAGE_H_
#define _STAGE_H_

#include "TypeDef.h"
#include "HaarArea.h"

class Stage
{
private:
public:
	double threshold;
	HaarArea* haarAreas;
	UInt numAreas;

	Stage(double threshold, HaarArea* areas) :
		threshold(threshold), haarAreas(areas)
	{

	}

	Stage() :
		threshold(0.0f), haarAreas(nullptr)
	{

	}

	~Stage()
	{
		delete[]haarAreas;
	}

	bool check(IntegralImage* image, UInt top, UInt left, double factor)
	{
		double value = 0;

		// For each feature in the feature tree of the current stage,
		for (UInt i = 0; i < this->numAreas; ++i)
		{
			// Evaluate the node's feature
			double sum = this->haarAreas[i].checkMatch(image, top, left, factor);

			// And increase the value accumulator
			if (sum < haarAreas[i].threshold * factor)
			{
				value += haarAreas[i].valueIfSmaller;
			}
			else
			{
				value += haarAreas[i].valueIfBigger;
			}
		}

		if (value < this->threshold)
		{
			return false;
		}

		return true;
	}
};

#endif