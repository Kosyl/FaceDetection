#include "stdafx.h"

#include <cstdlib>
#include <stdio.h>

#include <cv.h>
#include <highgui.h>

class ImgIO
{
public:
	ImgIO( void );
	~ImgIO( void );

	void ReadImgColor ( const char* path, unsigned char *&img );
	void WriteImgGrey ( const char* path, unsigned char *&img );
	void WriteImgColor( const char* path, unsigned char *&img );

	void ColorToGray( unsigned char *&imgColor, unsigned char *&imgGray );

	void OpenInputStream ( const char* path );
	void OpenOutputStream( const char* path );

	void ReadFromStream( unsigned char* &img );
	void WriteToStream ( unsigned char* &img );

	int GetXSize( void ) { return m_sizeX; }
	int GetYSize( void ) { return m_sizeY; }


private:

	void Mat2Char( unsigned char* &chImg, const cv::Mat &matImg );
	
	void Char2MatGrey ( unsigned char* &chImg, cv::Mat &matImg );
	void Char2MatColor( unsigned char* &chImg, cv::Mat &matImg );
	
	cv::VideoCapture m_vidInp;
	cv::VideoWriter m_vidOut;

	int m_sizeX;
	int m_sizeY;

};