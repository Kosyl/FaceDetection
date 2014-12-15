
#include "kernel.h"

// pamiec stala na informacje o malych fazach w duzej fazie
// "int" zamiast "HaarArea" z powodu ograniczen CUDA
// w kernelu rzutujemy to na HaarArea*
__constant__ int g_dev_areasInfo[220*(200/4)]; // = maks. liczba faz * zaokr¹glony w górê rozmiar HaarArea [B] / sizeof(int) [B]

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


void launchHaarKernel(UInt* image_dev, UInt* image2_dev, int numPhases, int xStep, int xEnd, int yStep, int yEnd, int stride, int windowSize, float invArea, bool* votingArea_dev, float threshold, HaarArea* areasStart, dim3 blockDim, dim3 gridDim)
{
	//kopiowanie informacji o strefach do pamieci stalej, dla wszystkich watkow
	checkCudaErrors(cudaMemcpyToSymbol(g_dev_areasInfo, areasStart, (numPhases)* sizeof(HaarArea), 0, cudaMemcpyHostToDevice));

	//statyczna wersja; WOLNA
	/*switch (numPhases)
	{
	case 3:
		haarKernelK<3> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 16:
		haarKernelK<16> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 21:
		haarKernelK<21> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 44:
		haarKernelK<44> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 33:
		haarKernelK<33> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 56:
		haarKernelK<56> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 51:
		haarKernelK<51> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 71:
		haarKernelK<71> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 80:
		haarKernelK<80> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 103:
		haarKernelK<103> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 111:
		haarKernelK<111> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 102:
		haarKernelK<102> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 135:
		haarKernelK<135> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 137:
		haarKernelK<137> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 140:
		haarKernelK<140> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 160:
		haarKernelK<160> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 177:
		haarKernelK<177> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 182:
		haarKernelK<182> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 211:
		haarKernelK<211> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	case 213:
		haarKernelK<213> << <gridDim, blockDim >> >(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
		break;
	default:
		return;
	}*/

	haarKernel<<<gridDim,blockDim>>>(image_dev, image2_dev, numPhases, xStep, xEnd, yStep, yEnd, stride, windowSize, invArea, votingArea_dev, threshold);
}

template<int K>
__global__ void haarKernelK(UInt* image_dev, UInt* image2_dev, int numPhases, int xStep, int xEnd, int yStep, int yEnd, int stride, int windowSize, float invArea, bool* votingArea_dev, float threshold)
{
	int xIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y*blockDim.y;

	int x = xIdx*xStep, y = yIdx*yStep;

	if (x >= xEnd || y >= yEnd)
		return;

	//stageSums_dev[y*stride + x] *= initMultiplier;
	float sum = 0.0f;

	float mean = getSumInRect(x, y, windowSize, windowSize, image_dev, stride) * invArea;
	float factor = getSum2InRect(x, y, windowSize, windowSize, image2_dev, stride) * invArea - (mean * mean);

	factor = (factor >= 0) ? sqrt(factor) : 1;
	float value = 0.0;

	HaarArea* areas = (HaarArea*)g_dev_areasInfo;
	HaarArea* currentArea = 0;
	HaarRectangle* currentRectangle = 0;

#pragma unroll
	for (int i = 0; i < K; ++i)
	{
		float sum = 0.0f;
		currentArea = &(areas[i]);

#pragma unroll
		for (int j = 0; j < 3; ++j)
		{
			currentRectangle = &(currentArea->rectangles[j]);
			sum += getSumInRect(x + currentRectangle->left*currentRectangle->sizeScale, y + currentRectangle->top*currentRectangle->sizeScale, currentRectangle->width*currentRectangle->sizeScale, currentRectangle->height*currentRectangle->sizeScale, image_dev, stride) * currentRectangle->scaledWeight* currentRectangle->valid;
		}

		// And increase the value accumulator
		if (sum < currentArea->threshold * factor)
		{
			value += currentArea->valueIfSmaller;
		}
		else
		{
			value += currentArea->valueIfBigger;
		}
	}
	sum += value;

	votingArea_dev[y*stride + x] &= sum > threshold;
}

__global__ void haarKernel(UInt* image_dev, UInt* image2_dev, int numPhases, int xStep, int xEnd, int yStep, int yEnd, int stride, int windowSize, float invArea, bool* votingArea_dev, float threshold)
{
	int xIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y*blockDim.y;

	// x,y w jednostkach mozliwych polozen okna, a nie pikseli! 
	int x = xIdx*xStep, y = yIdx*yStep;

	if (x >= xEnd || y >= yEnd)
		return;

	float mean = getSumInRect(x, y, windowSize, windowSize, image_dev, stride) * invArea;
	float factor = getSum2InRect(x, y, windowSize, windowSize, image2_dev, stride) * invArea - (mean * mean);
	factor = (factor >= 0) ? sqrt(factor) : 1;
	float value = 0.0;

	//rzutujemy pamiec stala na odpowiedni wskaznik
	HaarArea* areas = (HaarArea*)g_dev_areasInfo;
	
	for (int i = 0; i < numPhases; ++i)
	{
		float sum = 0.0f;
		HaarArea& currentArea=(areas[i]);

#pragma unroll
		for (int j = 0; j < 3; ++j)
		{
			HaarRectangle& currentRectangle=(currentArea.rectangles[j]);

			// do sumy dodajemy sume ppikseli w prostokacie
			// przemnoona najpierw przez jego wage
			// a potem przez pole "valid" - zerowe dla niektorych, zbednych prostokatow nr 3
			sum += getSumInRect(x + currentRectangle.left*currentRectangle.sizeScale, y + currentRectangle.top*currentRectangle.sizeScale, currentRectangle.width*currentRectangle.sizeScale, currentRectangle.height*currentRectangle.sizeScale, image_dev, stride) * currentRectangle.scaledWeight * currentRectangle.valid;
		}

		// zwiekszamy akumulator o odpowiednia wartosc
		if (sum < currentArea.threshold * factor)
		{
			value += currentArea.valueIfSmaller;
		}
		else
		{
			value += currentArea.valueIfBigger;
		}
	}

	//jesli value < threshold (duzej fazy), zerujemy okno tego watku - nie ma w nim twarzy
	votingArea_dev[y*stride + x] &= value > threshold;
}