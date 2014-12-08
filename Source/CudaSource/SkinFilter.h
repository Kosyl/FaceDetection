#include <cstdlib>

class SkinFilter
{
public:

	SkinFilter( void );
	~SkinFilter( void );

	void Filter( unsigned char *&img, unsigned char *&map, const int sizeX, const int sizeY );

private:
	void GenNgb( const int scale );
	int CalcScale(const int initScale, const int factor);

private: 
	int *m_ngb;

	int m_sizeX;
	int m_sizeY;
	int m_stride;
};