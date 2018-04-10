#include "stdafx.h"
#include "ReadTGA.h"

class Pixel {

public:
	int R, G, B, A;

	Pixel::Pixel() {};
	Pixel::Pixel(int r, int g, int b, int a) {
		this->A = a; this->R = r; this->G = g; this->B = b;
	}

};

struct TGAFILE
{
	unsigned char imageTypeCode;
	short int imageWidth;
	short int imageHeight;
	unsigned char bitCount;
	unsigned char *imageData;
	long imageSize;
	vector<Pixel> pixels;
	int colorMode;
};

void ReadTGA::LoadTGAFile(char *filename, TGAFILE *tgaFile)
{
	FILE *filePtr;
	unsigned char ucharBad;
	short int sintBad;
	long imageSize;
	int colorMode;
	unsigned char colorSwap;

	// Open the TGA file.
	fopen_s(&filePtr, filename, "rb");
	if (filePtr == NULL)
	{
		//return false;
	}

	// Read the two first bytes we don't need.
	fread(&ucharBad, sizeof(unsigned char), 1, filePtr);
	fread(&ucharBad, sizeof(unsigned char), 1, filePtr);

	// Which type of image gets stored in imageTypeCode.
	fread(&tgaFile->imageTypeCode, sizeof(unsigned char), 1, filePtr);

	// For our purposes, the type code should be 2 (uncompressed RGB image)
	// or 3 (uncompressed black-and-white images).
	if (tgaFile->imageTypeCode != 2 && tgaFile->imageTypeCode != 3)
	{
		fclose(filePtr);
		//return false;
	}

	// Read 13 bytes of data we don't need.
	fread(&sintBad, sizeof(short int), 1, filePtr);
	fread(&sintBad, sizeof(short int), 1, filePtr);
	fread(&ucharBad, sizeof(unsigned char), 1, filePtr);
	fread(&sintBad, sizeof(short int), 1, filePtr);
	fread(&sintBad, sizeof(short int), 1, filePtr);

	// Read the image's width and height.
	fread(&tgaFile->imageWidth, sizeof(short int), 1, filePtr);
	fread(&tgaFile->imageHeight, sizeof(short int), 1, filePtr);

	// Read the bit depth.
	fread(&tgaFile->bitCount, sizeof(unsigned char), 1, filePtr);

	// Read one byte of data we don't need.
	fread(&ucharBad, sizeof(unsigned char), 1, filePtr);

	// Color mode -> 3 = BGR, 4 = BGRA.
	colorMode = tgaFile->bitCount / 8;
	imageSize = tgaFile->imageWidth * tgaFile->imageHeight * colorMode;
	tgaFile->imageSize = imageSize;
	tgaFile->colorMode = colorMode;
	// Allocate memory for the image data.
	tgaFile->imageData = (unsigned char*)malloc(sizeof(unsigned char)*imageSize);

	// Read the image data.
	fread(tgaFile->imageData, sizeof(unsigned char), imageSize, filePtr);

	// Change from BGR to RGB so OpenGL can read the image data.
	vector<Pixel> pixels;
	for (int imageIdx = 0; imageIdx < imageSize; imageIdx += colorMode) {
		colorSwap = tgaFile->imageData[imageIdx];
		tgaFile->imageData[imageIdx] = tgaFile->imageData[imageIdx + 2];
		tgaFile->imageData[imageIdx + 2] = colorSwap;

		pixels.push_back(Pixel());
		pixels.back().R = tgaFile->imageData[imageIdx];
		pixels.back().G = tgaFile->imageData[imageIdx + 1];
		pixels.back().B = tgaFile->imageData[imageIdx + 2];
		pixels.back().A = 1;

		if (colorMode == 4) {
			pixels.back().A = tgaFile->imageData[imageIdx + 3];
		}
	}

	tgaFile->pixels = pixels;

	fclose(filePtr);
	//return true;
}
