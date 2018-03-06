#include "stdafx.h"
#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

class ReadTGA{
public:

	class Pixel {
		public:
			int R, G, B, A;
			Pixel::Pixel() {};
			Pixel::Pixel(int r, int g, int b, int a) {
				this->A = a; this->R = r; this->G = g; this->B = b;
			}
	};

	struct TGAFILE{
		unsigned char imageTypeCode;
		short int imageWidth;
		short int imageHeight;
		unsigned char bitCount;
		unsigned char *imageData;
		long imageSize;
		vector<Pixel> pixels;
		int colorMode;
	};

	static void LoadTGAFile(char *filename, TGAFILE *tgaFile);

};
