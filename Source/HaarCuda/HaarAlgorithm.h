#ifndef _HAAR_ALGORITHM_H_
#define _HAAR_ALGORITHM_H_

#include <vector>
#include "IntegralImage.h"
#include "Stage.h"

class HaarAlgorithm
{
public:
	HaarAlgorithm()
	{
		generateFaceStages();
	}
	~HaarAlgorithm()
	{
		delete[] this->stages;
	}
	std::vector<HaarRectangle> execute(IntegralImage* image);
protected:
private:
	Stage *stages;
	void generateFaceStages();
	bool searchWindow(IntegralImage* image, HaarRectangle window, double invArea);
};
#endif