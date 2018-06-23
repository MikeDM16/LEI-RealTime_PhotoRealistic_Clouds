#pragma once
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include "ReadTGA.h"
#include <stdlib.h>
#include "ClassicNoise.h"

class Noise{

private:
	static void generateNoise();
	static double smoothNoise(double x, double y);
	static double turbulence(double x, double y, double size);

public:	
	static void Noise::create_VTK();
	static void Noise::create_VTK(vector<ReadTGA::Pixel> p);
	static void Noise::lerFicheiro(char* dirFile);
};

