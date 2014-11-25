#include <cstdlib>

class FaceFilter
{
public:

	FaceFilter( void );
	~FaceFilter( void );

	void Filter( unsigned char *&img, unsigned char *&map, const int sizeX, const int sizeY );

private:
	void Close( unsigned char *&img, unsigned char *&out, const int sizeX, const int sizeY );
	void CreateStructElt( int *&elt, const int eltSize );

	void StretchColor( unsigned char *&img, const int sizeX, const int sizeY );
	void Negate( unsigned char *&img, unsigned char *&out, const int sizeX, const int sizeY );

	void Mask( unsigned char *&img, unsigned char *&mask, const int sizeX, const int sizeY );
	void FindHoles( unsigned char *&img, unsigned char *&holes, int x, int y, const int sizeX, const int sizeY );
	

};

