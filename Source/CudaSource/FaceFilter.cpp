
#include "FaceFilter.h"
#include "FaceFilterWrap.h"
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned char uchar;

FaceFilter::FaceFilter(void)
{

}

FaceFilter::~FaceFilter(void)
{

}

template <typename T>
T** getEmptyMatrix(int y, int x)
{
	T** tmp = new T*[y];
	for (int i = 0; i < y; ++i)
	{
		tmp[i] = new T[x];
		memset(tmp[i], 0, x*sizeof(T));
	}
	return tmp;
}
template <typename T>
void deleteMatrix(T**& matrix)
{
	if (matrix == nullptr)
		return;

	for (int i = 0; i < sizeof(matrix) / sizeof(T*); ++i)
	{
		if (matrix[i] != nullptr)
			delete[] matrix[i];
	}
	delete[] matrix;
	matrix = nullptr;
}

/// dilate i erode jest zaimpementowane w wersji GPU, trzeba tylko wyci¹gn¹c je do 
/// tego miejsca - tzn. napisaæ do nich wrapery
void erode(bool** mask, int width, int height, int maskSize = 3)
{
	int margin = (maskSize - 1) / 2;
	bool** result = getEmptyMatrix<bool>(height, width);
	for (int x = margin; x < width - margin; ++x)
	{
		for (int y = margin; y < height - margin; ++y)
		{
			bool pixel = true;
			for (int i = -margin; i <= margin; ++i)
			{
				for (int j = -margin; j <= margin; ++j)
				{
					pixel = pixel && mask[y + i][x + j];
				}
			}

			result[y][x] = pixel;
		}
	}
	for (int x = margin; x < width - margin; ++x)
	{
		for (int y = margin; y < height - margin; ++y)
		{
			mask[y][x] = result[y][x];
		}
	}
	deleteMatrix(result);
}

void dilate(bool** mask, int width, int height, int maskSize = 3)
{
	int margin = (maskSize - 1) / 2;
	bool** result = getEmptyMatrix<bool>(height, width);
	for (int x = margin; x < width - margin; ++x)
	{
		for (int y = margin; y < height - margin; ++y)
		{
			bool pixel = false;
			for (int i = -margin; i <= margin; ++i)
			{
				for (int j = -margin; j <= margin; ++j)
				{
					pixel = pixel || mask[y + i][x + j];
				}
			}

			result[y][x] = pixel;
		}
	}
	for (int x = margin; x < width - margin; ++x)
	{
		for (int y = margin; y < height - margin; ++y)
		{
			mask[y][x] = result[y][x];
		}
	}
	deleteMatrix(result);
}

int findAreas(bool** binPic, int** areas, int height, int width)
{
	unsigned int area = 1;
	unsigned int maxAreaIdx = 1;
	int etykieta = -1;
	for (int y = 0; y < height; ++y)
	{
		areas[y][0] = 0x7FFFFFFF;
		areas[y][width - 1] = 0x7FFFFFFF;
	}
	for (int x = 0; x < width; ++x)
	{
		areas[0][x] = 0x7FFFFFFF;
		areas[height - 1][x] = 0x7FFFFFFF;
	}

	for (int y = 1; y < height - 1; ++y)
	{
		for (int x = 1; x < width - 1; ++x)
		{
			if (binPic[y][x] == 1)
			{
				area = std::max(binPic[y][x - 1], std::max(binPic[y - 1][x - 1], std::max(binPic[y - 1][x], binPic[y - 1][x + 1])));
				if (area == 1)
				{
					areas[y][x] = std::min(areas[y][x - 1], std::min(areas[y - 1][x - 1], std::min(areas[y - 1][x], areas[y - 1][x + 1])));;
				}
				else
				{
					areas[y][x] = maxAreaIdx++;
				}
			}
			else
			{
				areas[y][x] = 0x7FFFFFFF;
			}
		}
	}
	bool changed = true;
	while (changed)
	{
		changed = false;
		for (int y = 1; y < height - 1; ++y)
		{
			for (int x = 1; x < width - 1; ++x)
			{
				if (binPic[y][x] != 0)
				{
					area = std::min(areas[y][x - 1],
						std::min(areas[y - 1][x - 1],
						std::min(areas[y - 1][x],
						std::min(areas[y - 1][x + 1],
						std::min(areas[y][x + 1],
						std::min(areas[y + 1][x + 1],
						std::min(areas[y + 1][x], areas[y + 1][x - 1])))))));

					if (area < areas[y][x])
					{
						areas[y][x] = area;
						changed = true;
					}
				}
			}
		}
	}

	return maxAreaIdx;
}

std::vector<HaarRectangle> FaceFilter::FindFaces(unsigned char *&img, unsigned char *&out, const int sizeX, const int sizeY)
{
	std::vector<HaarRectangle> result;
	for (int i = 0; i < sizeX * sizeY; i++)
	{
		if (!img[i])
			img[i] = 0;
		else
			img[i] = 255;
	}

	bool** map = getEmptyMatrix<bool>(sizeY, sizeX);
	bool** holes = getEmptyMatrix<bool>(sizeY, sizeX);

	for (int i = 0; i < sizeX; ++i)
	{
		for (int j = 0; j < sizeY; ++j)
		{
			map[j][i] = img[j * sizeX + i] != 0;
		}
	}

	int** areas = getEmptyMatrix<int>(sizeY, sizeX);

	dilate(map, sizeX, sizeY);
	dilate(map, sizeX, sizeY);
	erode(map, sizeX, sizeY);
	erode(map, sizeX, sizeY);

	for (int i = 0; i < sizeX; ++i)
	{
		for (int j = 0; j < sizeY; ++j)
		{
			holes[j][i] = img[j * sizeX + i] == 0 && map[j][i];
		}
	}

	int numAreas = findAreas(map, areas, sizeY, sizeX);

	erode(holes, sizeX, sizeY);

	for (int i = 0; i < sizeX; ++i)
	{
		for (int j = 0; j < sizeY; ++j)
		{
			img[j * sizeX + i] = 0;
		}
	}

	for (int i = 0; i < sizeX; ++i)
	{
		for (int j = 0; j < sizeY; ++j)
		{
			if (holes[j][i] != 0)
			{
				int etykieta = areas[j][i];
				HaarRectangle res;
				res.top = res.left = 0xFFFFFFFF;
				res.bottom = res.right = 0;
				for (UInt q = 0; q < sizeX; q++)
				{
					for (UInt p = 0; p < sizeY; ++p)
					{
						if (areas[p][q] == etykieta)
						{
							res.top = std::min(p, res.top);
							res.left = std::min(q, res.left);
							res.bottom = std::max(p, res.bottom);
							res.right = std::max(q, res.right);
						}
					}
				}
				res.height = res.bottom - res.top;
				res.width = res.right - res.left;
				result.push_back(res);
			}
		}
	}


	deleteMatrix(map);
	deleteMatrix(holes);
	deleteMatrix(areas);

	return result;
}

std::vector<HaarRectangle> FaceFilter::Filter(unsigned char *&img, unsigned char *&map, const int sizeX, const int sizeY)
{
	assert(sizeX);
	assert(sizeY);

	uchar *closed;
	uchar *stretched;
	uchar *holes;

	const int stride = sizeX*sizeY;
	const int eltSize = 7;

	uchar* mapDevice;
	uchar* imgDevice;
	uchar* closedDevice;

	cudaMalloc(&mapDevice, sizeof(uchar) * sizeX * sizeY);
	cudaMalloc(&imgDevice, sizeof(uchar) * sizeX * sizeY);
	cudaMalloc(&closedDevice, sizeof(uchar) * sizeX * sizeY);

	cudaMemcpy(mapDevice, map, stride*sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(imgDevice, img, stride*sizeof(uchar), cudaMemcpyHostToDevice);

	CloseDevice(mapDevice, closedDevice, eltSize, sizeX, sizeY);

	MaskDevice(imgDevice, closedDevice, sizeX, sizeY);
	StretchDevice(imgDevice, sizeX, sizeY);
	
	/// do debugowania, te wartoœci s¹ rysowane w DCIDetect.Run()
	cudaMemcpy(map, imgDevice, sizeof(uchar)*stride, cudaMemcpyDeviceToHost);
	
	// do tego miejsca liczy GPU
	cudaMemcpy(img, imgDevice, sizeof(uchar)*stride, cudaMemcpyDeviceToHost);
	std::vector<HaarRectangle> result = FindFaces(img, img, sizeX, sizeY);

	cudaFree(mapDevice);
	cudaFree(imgDevice);
	cudaFree(closedDevice);
	
	return result;
}

/// to co jest poni¿ej nie jest ju¿ wykorzysytwane w implementacji GPU
#if 0
void FaceFilter::Mask(unsigned char *&img, unsigned char *&mask, const int sizeX, const int sizeY)
{
	assert(img);

	for (int i = 0; i < sizeX * sizeY; i++)
	{
		if (mask[i] == 0)
			img[i] = 0;
	}
}

void FaceFilter::Close(unsigned char *&img, unsigned char *&out, const int sizeX, const int sizeY)
{
	const int eltSize = 7;

	int   *structElt;
	uchar *tempImg;
	structElt = new int[2 * eltSize * eltSize];
	tempImg = new uchar[sizeX * sizeY];

	CreateStructElt(structElt, eltSize);

	//dilate
	for (int i = 0; i < sizeX; i++)
	{
		for (int j = 0; j < sizeY; j++)
		{
			if (i < eltSize || i > sizeX - eltSize || j < eltSize || j > sizeY - eltSize)
			{
				tempImg[i + sizeX * j] = 0;
			}
			else
			{
				int val = 0;

				for (int k = 0; k < 2 * eltSize * eltSize; k += 2)
				{
					val += img[i + structElt[k] + (sizeX * (j + structElt[k + 1]))];
				}

				if (val)
					val = 255;

				for (int k = 0; k < 2 * eltSize * eltSize; k += 2)
				{
					tempImg[i + structElt[k] + (sizeX * (j + structElt[k + 1]))] = val;
				}
			}
		}
	}

	//erode
	for (int i = 0; i < sizeX; i++)
	{
		for (int j = 0; j < sizeY; j++)
		{
			uchar val = 0;

			if (i < eltSize || i > sizeX - eltSize || j < eltSize || j > sizeY - eltSize)
			{
				out[i + sizeX * j] = 0;
			}
			else
			{
				for (int k = 0; k < 2 * eltSize * eltSize; k += 2)
				{
					if (tempImg[i + structElt[k] + (sizeX * (j + structElt[k + 1]))] == 0)
					{
						val = 0;
						break;
					}

					val = 255;
				}

				for (int k = 0; k < 2 * eltSize * eltSize; k += 2)
				{
					out[i + structElt[k] + (sizeX * (j + structElt[k + 1]))] = val;
				}
			}
		}
	}

	delete[] structElt;
	delete[] tempImg;
}

void FaceFilter::CreateStructElt(int *&elt, const int eltSize)
{
	const int factor = (eltSize - 1) / 2;
	int currPos = 0;

	std::cout << "Generating ngb: \n";

	for (int i = -factor; i <= factor; i++)
	{
		for (int j = -factor; j <= factor; j++)
		{
			elt[currPos] = i;
			elt[currPos + 1] = j;

			currPos += 2;
		}
	}
}

void FaceFilter::StretchColor(unsigned char *&img, const int sizeX, const int sizeY)
{
	uchar maxVal = 0;
	// uchar minVal = 255;

	for (int i = 0; i < sizeX * sizeY; i++)
	{
		const int pxlVal = (int)img[i];

		/*if( pxlVal < minVal )
			minVal = pxlVal;*/

		if (pxlVal > maxVal)
			maxVal = pxlVal;
	}

	for (int i = 0; i < sizeX * sizeY; i++)
	{
		const int newVal = (((float)img[i] / (float)maxVal) * 255.f);

		if (newVal > 95 && newVal < 240)
			img[i] = newVal;
		else
			img[i] = 0;
	}

}


void FaceFilter::FindHoles(unsigned char *&img, unsigned char *&holes, int x, int y, const int sizeX, const int sizeY)
{
	if (x < 0 || x >= sizeX || y < 0 || y >= sizeY)
		return;

	holes[x + sizeX * y] = 0;

	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			if (i == 0 && j == 0)
				continue;
			if (x + i < 0 || x + i >= sizeX || y + j < 0 || y + j >= sizeY)
				continue;
			if (img[(x + i) + sizeX * (y + j)] == 255 && holes[(x + i) + sizeX * (y + j)] == 255)
			{
				FindHoles(img, holes, x + i, y + j, sizeX, sizeY);
			}
		}
	}
}

#endif