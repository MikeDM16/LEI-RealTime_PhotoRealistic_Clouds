#include "stdafx.h"
#pragma once
#include "Noise.h"
#include <string>

#define noiseWidth 256
#define noiseHeight 256

double noise[noiseHeight][noiseWidth]; //the noise array

void Noise::generateNoise()
{
	for (int y = 0; y < noiseHeight; y++)
		for (int x = 0; x < noiseWidth; x++)
		{
			noise[y][x] = (rand() % 32768) / 32768.0;
		}
}

double Noise::smoothNoise(double x, double y)
{
	//get fractional part of x and y
	double fractX = x - int(x);
	double fractY = y - int(y);

	//wrap around
	int x1 = (int(x) + noiseWidth) % noiseWidth;
	int y1 = (int(y) + noiseHeight) % noiseHeight;

	//neighbor values
	int x2 = (x1 + noiseWidth - 1) % noiseWidth;
	int y2 = (y1 + noiseHeight - 1) % noiseHeight;

	//smooth the noise with bilinear interpolation
	double value = 0.0;
	value += fractX     * fractY     * noise[y1][x1];
	value += (1 - fractX) * fractY     * noise[y1][x2];
	value += fractX     * (1 - fractY) * noise[y2][x1];
	value += (1 - fractX) * (1 - fractY) * noise[y2][x2];

	return value;
}

double Noise::turbulence(double x, double y, double size)
{
	double value = 0.0, initialSize = size;

	while (size >= 1)
	{
		value += smoothNoise(x / size, y / size) * size;
		size /= 2.0;
	}

	return(128.0 * value / initialSize);
}

void Noise::lerFicheiro(char* dirFile) {

	cout << "Sera usado o ficheiro na diretoria:\n " << dirFile << endl;
	FILE *fd;
	fopen_s(&fd, dirFile, "r");
	if (!fd) {
		cout << "Nao foi encontrado nenhum ficheiro na diretoria " << dirFile << endl;
	}
	else {
		std::ofstream vtkstream;
		vtkstream.open("foot.vtk", std::ios::out | std::ios::app | std::ios::binary);
		if (vtkstream) {
			vtkstream << "# vtk DataFile Version 3.0" << "\n";
			vtkstream << "Exemple teste" << "\n";
			vtkstream << "BINARY" << "\n";
			vtkstream << "DATASET STRUCTURED_POINTS" << std::endl;
			vtkstream << "DIMENSIONS 32 32 32" << std::endl;
			vtkstream << "ORIGIN 0 0 0" << std::endl;
			vtkstream << "SPACING 0.5 0.5 0.5" << std::endl;
			vtkstream << "POINT_DATA 32768" << std::endl;
			vtkstream << "SCALARS image_data float" << std::endl;
			vtkstream << "LOOKUP_TABLE default" << std::endl;
		}
		char *linha = new char[100];
		int i = 0;
		while (fgets(linha, 100, fd)) {
			printf("%d\n", i++);
			float m = abs(atof(linha));
			vtkstream.write((char*)&m, sizeof(float));
		}
		vtkstream.close();
	}
	fclose(fd);
}

void Noise::create_VTK(){
	std::ofstream vtkstream;
	vtkstream.open("foot.vtk", std::ios::out | std::ios::app | std::ios::binary);
	if (vtkstream) {
		vtkstream << "# vtk DataFile Version 3.0" << "\n";
		vtkstream << "Exemple teste" << "\n";
		vtkstream << "BINARY" << "\n";
		vtkstream << "DATASET STRUCTURED_POINTS" << std::endl;
		vtkstream << "DIMENSIONS 256 128 256" << std::endl;
		vtkstream << "ORIGIN 0 0 0" << std::endl;
		vtkstream << "SPACING 0.5 0.5 0.5" << std::endl;
		vtkstream << "POINT_DATA 8388608" << std::endl;
		vtkstream << "SCALARS image_data float" << std::endl;
		vtkstream << "LOOKUP_TABLE default" << std::endl;
		
		Noise::generateNoise();

		/*for (unsigned int z = 0; z < 256; z++) {

			for (unsigned int y = 0; y < 64; y++) {
				for (unsigned int x = 0; x < 64; x++) {
					float valor = 1;
					vtkstream.write((char*)&valor, sizeof(float));
				}
			}
			for (unsigned int y = 0; y < 64; y++) {
				for (unsigned int x = 0; x < 64; x++) {
					float valor = 0;
					vtkstream.write((char*)&valor, sizeof(float));
				}
			}
			for (unsigned int y = 0; y < 64; y++) {
				for (unsigned int x = 0; x < 64; x++) {
					float valor = 1;
					vtkstream.write((char*)&valor, sizeof(float));
				}
			}
			for (unsigned int y = 0; y < 64; y++) {
				for (unsigned int x = 0; x < 64; x++) {
					float valor = 0;
					vtkstream.write((char*)&valor, sizeof(float));
				}
			}
		}*/

		//ATEMPT NUMBER 1M

		//for (unsigned int z = 0; z < 256; z++) {
		//	for (unsigned int x = 0; x < 256; x++) {
		//		for (unsigned int y = 0; y < 256; y++) {
		//			//float valor = (float)( turbulence(x, y, 64)/256.f);
		//			double valor = ClassicNoise::noise(x % 256, y % 256, z % 256);
		//			if(valor != 0 )
		//				std::cout << valor << std::endl;					
		//			vtkstream.write((char*)&valor, sizeof(float));
		//		}
		//	}
		//}

		vtkstream.close();
	}
	else {
		std::cout << "ERROR" << std::endl;
	}
}

void Noise::create_VTK(vector<ReadTGA::Pixel> pixels) {
	std::ofstream vtkstream;
	vtkstream.open("foot.vtk", std::ios::out | std::ios::app | std::ios::binary);
	if (vtkstream) {
		vtkstream << "# vtk DataFile Version 3.0" << "\n";
		vtkstream << "Exemple teste" << "\n";
		vtkstream << "BINARY" << "\n";
		vtkstream << "DATASET STRUCTURED_POINTS" << std::endl;
		vtkstream << "DIMENSIONS 256 256 256" << std::endl;
		vtkstream << "ORIGIN 0 0 0" << std::endl;
		vtkstream << "SPACING 0.5 0.5 0.5" << std::endl;
		vtkstream << "POINT_DATA 16777216" << std::endl;
		vtkstream << "SCALARS image_data float" << std::endl;
		vtkstream << "LOOKUP_TABLE default" << std::endl;

		for (ReadTGA::Pixel p : pixels) {
			float valor = p.R;
			vtkstream.write((char*)&valor, sizeof(float));
			valor = p.R;
			vtkstream.write((char*)&valor, sizeof(float));
			valor = p.G;
			vtkstream.write((char*)&valor, sizeof(float));
			valor = p.B;
			vtkstream.write((char*)&valor, sizeof(float));
			valor = p.A;
			vtkstream.write((char*)&valor, sizeof(float));
		}
		vtkstream.close();
		std::cout << "Fim" << std::endl;
	}
	else {
		std::cout << "ERROR" << std::endl;
	}
}

