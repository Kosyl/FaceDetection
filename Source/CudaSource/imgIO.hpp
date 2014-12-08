#ifndef _IMGIO_HPP_
#define _IMGIO_HPP_

#include "cudafun.cuh"

#include <cstdlib>
#include <stdio.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "TypeDef.h"

class ImgIO
{
public:
	ImgIO( void );
	~ImgIO( void );

	void ReadImgColor ( const char* path, unsigned char *&img );
	void WriteImgGrey ( const char* path, unsigned char *&img );
	void WriteImgColor( const char* path, unsigned char *&img );

	void ColorToGray(unsigned char *&imgColor, unsigned char *&imgGray);
	void ColorToRed(unsigned char *&imgColor, unsigned char *&imgRed);
	void DeviceColorToGray( unsigned char *&imgColor, unsigned char *&imgGray );

	void Transpose(unsigned char *&source, unsigned char *&dst);

	void OpenInputStream ( const char* path );
	void OpenOutputStream( const char* path );

	void ReadFromStream( unsigned char* &img );
	void WriteToStream ( unsigned char* &img );

	UInt getSizeX()
	{
		return m_sizeX;
	}

	UInt getSizeY()
	{
		return m_sizeY;
	}

private:

	void Mat2Char( unsigned char* &chImg, const cv::Mat &matImg );
	
	void Char2MatGrey ( unsigned char* &chImg, cv::Mat &matImg );
	void Char2MatColor( unsigned char* &chImg, cv::Mat &matImg );
	
	cv::VideoCapture m_vidInp;
	cv::VideoWriter m_vidOut;

	int m_sizeX;
	int m_sizeY;

};

#endif