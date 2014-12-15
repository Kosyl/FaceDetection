
#include "kernel.h"

__device__ long getSumInRect(UInt left, UInt top, UInt windowWidth, UInt windowHeight, UInt* values, int stride)
{
	int a = stride * (top)+(left);
	int b = stride * (top + windowHeight) + (left + windowWidth);
	int c = stride * (top + windowHeight) + (left);
	int d = stride * (top)+(left + windowWidth);

	return values[a] + values[b] - values[c] - values[d];
}

__device__ long getSum2InRect(UInt left, UInt top, UInt windowWidth, UInt windowHeight, UInt* values2, int stride)
{
	int a = stride * (top)+(left);
	int b = stride * (top + windowHeight) + (left + windowWidth);
	int c = stride * (top + windowHeight) + (left);
	int d = stride * (top)+(left + windowWidth);

	return values2[a] + values2[b] - values2[c] - values2[d];
}


void launchHaarKernel(UInt* image_dev, UInt* image2_dev, float* stageSums_dev, HaarArea* areas_dev, int numPhases, int xStep, int xEnd, int yStep, int yEnd, int stride, int windowSize, float invArea, dim3 blockDim, dim3 gridDim)
{
	haarKernel<<<gridDim,blockDim>>>(image_dev, image2_dev, stageSums_dev, areas_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea);
}


__global__ void haarKernel(UInt* image_dev, UInt* image2_dev, float* stageSums_dev, HaarArea* areas_dev, int numPhases, int xStep, int xEnd, int yStep, int yEnd, int stride, int windowSize, float invArea)
{
	int xIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y*blockDim.y;

	int x = xIdx*xStep, y = yIdx*yStep;

	if (x >= xEnd || y >= yEnd)
		return;

	float mean = getSumInRect(x, y, windowSize, windowSize, image_dev, stride) * invArea;
	float factor = getSum2InRect(x, y, windowSize, windowSize, image2_dev, stride) * invArea - (mean * mean);

	factor = (factor >= 0) ? sqrt(factor) : 1;
	float value = 0.0;
	for (int i = 0; i < numPhases; ++i)
	{
		float sum = 0.0f;
		for (int j = 0; j < areas_dev[i].numRectangles; ++j)
		{
			sum += getSumInRect(x + areas_dev[i].rectangles[j].left*areas_dev[i].rectangles[j].sizeScale, y + areas_dev[i].rectangles[j].top*areas_dev[i].rectangles[j].sizeScale, areas_dev[i].rectangles[j].width*areas_dev[i].rectangles[j].sizeScale, areas_dev[i].rectangles[j].height*areas_dev[i].rectangles[j].sizeScale, image_dev, stride) * areas_dev[i].rectangles[j].scaledWeight;
		}

		// And increase the value accumulator
		if (sum < areas_dev[i].threshold * factor)
		{
			value += areas_dev[i].valueIfSmaller;
		}
		else
		{
			value += areas_dev[i].valueIfBigger;
		}
	}
	stageSums_dev[y*stride + x] += value;
}