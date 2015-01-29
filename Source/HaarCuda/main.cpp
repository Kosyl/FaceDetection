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
#include <cv.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

typedef LARGE_INTEGER app_timer_t;

#define timer(t_ptr) QueryPerformanceCounter(t_ptr)

//zaznaczenie wyniku na obrazie w skali szarosci
void mark(HaarRectangle& rect, unsigned char* imgGrey, UInt stride)
{
	for (UInt i = 0; i < rect.height; ++i)
	{
		imgGrey[(rect.top + i)*stride + rect.left] = 250;
		imgGrey[(rect.top + i)*stride + rect.left + rect.width] = 250;
	}
	for (UInt i = 0; i < rect.width; ++i)
	{
		imgGrey[(rect.top*stride) + rect.left + i] = 250; // gora
		imgGrey[((rect.top + rect.height)*stride) + rect.left + i] = 250; //dol
	}
}

int main(int argc, char *argv[])
{
	int xx = 20 >> 3;
	++xx;

	/*if (argc != 2)
	{
	printf("uzycie: %s nazwa_pliku_zdjecia.jpg", argv[0]);
	return 0;
	}
	*/
	int x = sizeof(HaarArea);
	x = sizeof(float);


	app_timer_t start, stop;
	const char* pathIn = "D:\\Studia\\RIM\\projekt\\RIM\\1.jpg";//argv[1];
	const char* pathOutGrey = "wynik.jpg";

	unsigned char *imgColor = NULL;
	unsigned char *imgGrey = NULL;
	unsigned char *transposed = NULL;


	checkCudaErrors(cudaSetDevice(0));

	ImgIO imgIO;

	imgIO.ReadImgColor(pathIn, imgColor);

	imgIO.ColorToGray(imgColor, imgGrey);

	HaarAlgorithm alg;

	timer(&start);
	std::vector<HaarRectangle> result = alg.execute(imgIO.getSizeX(), imgIO.getSizeY(), imgGrey);
	timer(&stop);

	double etime;
	LARGE_INTEGER clk_freq;
	QueryPerformanceFrequency(&clk_freq);
	etime = (stop.QuadPart - start.QuadPart) / (double)clk_freq.QuadPart;
	printf("time = %.3f ms\n", etime * 1e3);
	for (std::vector<HaarRectangle>::iterator i = result.begin(); i != result.end(); ++i)
	{
		mark(*i, imgGrey, imgIO.getSizeX());
	}

	imgIO.WriteImgGrey(pathOutGrey, imgGrey);

	delete[] imgColor;
	delete[] imgGrey;
	delete[] transposed;

	if (IsDebuggerPresent()) getchar();

	return 0;
}