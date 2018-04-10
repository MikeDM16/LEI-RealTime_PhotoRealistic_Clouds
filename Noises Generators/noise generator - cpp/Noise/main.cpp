#include "stdafx.h"
#pragma once
#include <stdio.h>     /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>
#include "ReadTGA.h"
#include "Noise.h"

using namespace std;

int main() {
	 
	char* filename[] = { "noiseerosion.tga", "noiseerosionpacked.tga", "noiseshape.tga", "noiseshapepacked.tga" };

	struct ReadTGA::TGAFILE* tga_file = (struct ReadTGA::TGAFILE*) malloc(sizeof(struct ReadTGA::TGAFILE));
	ReadTGA::LoadTGAFile(filename[1], tga_file);

	std::cout << filename[1] << ": " << std::endl;
	std::cout << "	altura " << tga_file->imageHeight << std::endl;
	std::cout << "	largura " << tga_file->imageWidth << std::endl;

	//vector<ReadTGA::Pixel> p = tga_file->pixels; 

	////Noise::Create_vtk(p);

	std::cout << "fim \n";
	free(tga_file);

	Noise::lerFicheiro("vtk_values");
	return 0;
}