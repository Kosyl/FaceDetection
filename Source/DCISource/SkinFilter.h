#include <cstdlib>

class SkinFilter
{
public:

	SkinFilter( void );
	~SkinFilter( void );

	void Filter( unsigned char *&img, unsigned char *&map, const int sizeX, const int sizeY );

private:
	
	void RGB2IRB( unsigned char *&imgIn, float *&imgOut );
	
	void  MedianFilter( float *&img, float *&medianImg, int scale );
	float GetMedian( float *& img, int scale );
	
	void Texture( float *&texture, float *&img, float *&imgFiltered );
	void HueSaturation( float *&hue, float *&saturation, float *& Rg, float *&Rb );
	
	void GenerateMap( float *& medianImg, float *& hue, float *& saturation, unsigned char *& map );
	void WriteMap( unsigned char *&map );

	void  GenNgb( const int scale );

private: 
	int *m_ngb;

	int m_sizeX;
	int m_sizeY;
	int m_stride;
};