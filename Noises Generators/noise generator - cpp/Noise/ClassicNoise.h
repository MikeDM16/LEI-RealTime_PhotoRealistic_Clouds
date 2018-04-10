#pragma once
#include "stdafx.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>

class ClassicNoise {
	// Classic Perlin noise in 3D, for comparison
	public:
		// This method is a *lot* faster than using (int)Math.floor(x)
		static int fastfloor(double x);

		static double dot(int g[], double x, double y, double z);
		static double mix(double a, double b, double t);
		static double fade(double t);
		static double noise(double x, double y, double z);
};