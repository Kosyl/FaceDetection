#include <cstdio>
#include <stdio.h>
#include "HaarAlgorithm.h"
#include "IntegralImage.h"
#include "imgIO.hpp"
#include <vector>
#include <windows.h>
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "cuda.h"

typedef LARGE_INTEGER app_timer_t;

#define timer(t_ptr) QueryPerformanceCounter(t_ptr)

void mark(HaarRectangle& rect, unsigned char* imgGrey, UInt stride)
{
	for (int i = 0; i < rect.height; ++i)
	{
		imgGrey[(rect.top + i)*stride + rect.left] = 250;
		imgGrey[(rect.top + i)*stride + rect.left + rect.width] = 250;
	}
	for (int i = 0; i < rect.width; ++i)
	{
		imgGrey[(rect.top*stride) + rect.left + i] = 250; // gora
		imgGrey[((rect.top + rect.height)*stride) + rect.left + i] = 250; //dol
	}
}

int main()
{
	app_timer_t start, stop;
	const char* pathIn = "../zdj.jpg";
	const char* pathOutGrey = "../obrazTestOutGrey.jpg";
	const char* pathOutColor = "../obrazTestOutColor.jpg";

	unsigned char *imgColor = NULL;
	unsigned char *imgGrey = NULL;
	unsigned char *imgRed = NULL;
	unsigned char *transposed = NULL;

	checkCudaErrors(cudaSetDevice(0));

	ImgIO imgIO;

	imgIO.ReadImgColor(pathIn, imgColor);

	imgIO.ColorToRed(imgColor, imgRed);
	imgIO.ColorToGray(imgColor, imgGrey);
	//imgIO.Transpose(imgGrey, transposed);

	IntegralImage image(imgIO.getSizeX(), imgIO.getSizeY(), imgGrey);
	HaarAlgorithm alg;

	timer(&start);
	std::vector<HaarRectangle> result = alg.execute(&image);
	timer(&stop);

	double etime;
	LARGE_INTEGER clk_freq;
	QueryPerformanceFrequency(&clk_freq);
	etime = (stop.QuadPart - start.QuadPart) /
		(double)clk_freq.QuadPart;
	printf("time = %.3f ms\n", etime * 1e3);

	for (std::vector<HaarRectangle>::iterator i = result.begin(); i != result.end(); ++i)
	{
		mark(*i, imgGrey, imgIO.getSizeX());
	}

	imgIO.WriteImgColor(pathOutColor, imgColor);
	imgIO.WriteImgGrey(pathOutGrey, imgGrey);

	delete[] imgColor;
	delete[] imgGrey;
	delete[] imgRed;
	delete[] transposed;


	if (IsDebuggerPresent()) getchar();

	return 0;
}