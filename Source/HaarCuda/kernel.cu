
#include "kernel.h"

// pamiec stala na informacje o malych fazach w duzej fazie
// "int" zamiast "HaarArea" z powodu ograniczen CUDA
// w kernelu rzutujemy to na HaarArea*
__constant__ UInt g_dev_areasPerStage[STAGES_COUNT];
__constant__ float g_dev_thresholdPerStage[STAGES_COUNT];

__device__ long getSumInRect(UInt left, UInt top, UInt windowWidth, UInt windowHeight, UInt* values, int stride)
{
	int a = stride * (top)+(left);
	int b = stride * (top + windowHeight) + (left + windowWidth);
	int c = stride * (top + windowHeight) + (left);
	int d = stride * (top)+(left + windowWidth);

	return values[a] + values[b] - values[c] - values[d];
}


void launchHaarKernel(UInt* image_dev, float* weights_dev, bool* votingArea_dev, UInt imageStride, UInt weightsStride, HaarArea* allAreas_dev, UInt areasInStage[STAGES_COUNT], float thresholds[STAGES_COUNT], dim3 blockDim, dim3 gridDim)
{
	//kopiowanie informacji o strefach do pamieci stalej, dla wszystkich watkow
	static bool areasCopied = false;
	if (!areasCopied)
	{
		checkCudaErrors(cudaMemcpyToSymbol(g_dev_areasPerStage, areasInStage, (STAGES_COUNT)* sizeof(UInt), 0, cudaMemcpyHostToDevice));
		areasCopied = true;
	}
	checkCudaErrors(cudaMemcpyToSymbol(g_dev_thresholdPerStage, thresholds, (STAGES_COUNT)* sizeof(float), 0, cudaMemcpyHostToDevice));

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

	size_t shSize = (TILE_SIZE_IN_PIX + 1)*(TILE_SIZE_IN_PIX + 1)*sizeof(UInt);

	haarKernel << <gridDim, blockDim, shSize >> >(image_dev, weights_dev, votingArea_dev, imageStride, weightsStride, allAreas_dev);
}

__global__ void haarKernel(UInt* image_dev, float* weights_dev, bool* votingArea_dev, UInt imageStride, UInt weightsStride, HaarArea* allAreas_dev)
{
	extern __shared__ UInt sh_mem[];

	//float* factor = &sh_mem[0];
	UInt *roi = (UInt*)&sh_mem[0];

	size_t blockInPictureX = blockIdx.x * TILE_SIZE;
	size_t blockInPictureY = blockIdx.y * TILE_SIZE;

	size_t windowInPictureX = threadIdx.x + blockIdx.x*blockDim.x;
	size_t windowInPictureY = threadIdx.y + blockIdx.y*blockDim.y;
	float factor = weights_dev[windowInPictureY*weightsStride + windowInPictureX];

	size_t imagePxlX = windowInPictureX * WINDOW_STEP_IN_PIX;
	size_t imagePxlY = windowInPictureY * WINDOW_STEP_IN_PIX;

	size_t x, y;

	if (blockIdx.x == 0 && blockIdx.y == 0)
	for (x = windowInPictureX * WINDOW_STEP_IN_PIX; x < TILE_SIZE_IN_PIX-1; x += blockDim.x*WINDOW_STEP_IN_PIX)
	{
		for (y = windowInPictureY * WINDOW_STEP_IN_PIX; y < TILE_SIZE_IN_PIX-1; y += blockDim.y*WINDOW_STEP_IN_PIX)
		{
			for (size_t dy = 0; dy < WINDOW_STEP_IN_PIX; ++dy)
			{
				for (size_t dx = 0; dx < WINDOW_STEP_IN_PIX; ++dx)
				{
					size_t roiY = (y + dy);
					size_t roiX = x + dx;
					size_t picY = (blockInPictureY + (y + dy));
					size_t picX = (blockInPictureX + (x + dx));

					if (roiY == 2 && roiX == 0)
						picX = picX;
					roi[roiY*TILE_SIZE_IN_PIX + (roiX)] = image_dev[picY*imageStride + picX];
				}
			}
		}
	}

	__syncthreads();

	x = threadIdx.x*WINDOW_STEP_IN_PIX;
	y = threadIdx.y*WINDOW_STEP_IN_PIX;

	//rzutujemy pamiec stala na odpowiedni wskaznik
	HaarArea* currentArea = allAreas_dev;
	UInt areasInCurrentStageCount;

	bool foundFace = true;

	for (size_t stageIdx = 0; stageIdx < STAGES_COUNT; ++stageIdx)
	{
		float stageSum = 0.0;

		areasInCurrentStageCount = g_dev_areasPerStage[stageIdx];
		for (size_t areaIdx = 0; areaIdx < areasInCurrentStageCount; ++areaIdx)
		{
			float areaSum = 0.0f;

#pragma unroll
			for (int j = 0; j < 3; ++j)
			{
				HaarRectangle& currentRectangle = (currentArea->rectangles[j]);

				int a = TILE_SIZE_IN_PIX * (y + currentRectangle.top) + (x + currentRectangle.left);
				int b = TILE_SIZE_IN_PIX * (y + currentRectangle.top + currentRectangle.height) + (x + currentRectangle.left + currentRectangle.width);
				int c = TILE_SIZE_IN_PIX * (y + currentRectangle.top + currentRectangle.height) + (x + currentRectangle.left);
				int d = TILE_SIZE_IN_PIX * (y + currentRectangle.top) + (x + currentRectangle.left + currentRectangle.width);

				areaSum += (roi[a] + roi[b] - roi[c] - roi[d]) * currentRectangle.weight * currentRectangle.valid;
			}

			stageSum += (currentArea->valueIfSmaller)*(areaSum < currentArea->threshold * factor) + (currentArea->valueIfBigger)*(areaSum >= currentArea->threshold * factor);

			currentArea = currentArea + 1;
		}

		foundFace &= stageSum > g_dev_thresholdPerStage[stageIdx];

		if (!foundFace)
		{
			votingArea_dev[windowInPictureY*weightsStride + windowInPictureX] = false;
			return;
		}
	}
}