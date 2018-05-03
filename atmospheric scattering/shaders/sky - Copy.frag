#version 430

uniform vec3 camUp = vec3(0,1,0);
uniform vec3 camView = vec3(0,0,-1);
uniform float ratio = 1;
uniform float fov = 65;

uniform int divisions = 32;
uniform int divisonsLightRay = 32;
uniform int cameraMode = 1;
uniform float exposure = 1.5;


 
uniform float betaMf = 21.0e-6f;
//uniform vec3 betaM = vec3(21.0e-6f);
uniform float Hr = 7994;
uniform float Hm = 1200;
uniform float g = 0.99;
uniform vec2 sunAngles = vec2(28,28);

uniform int wavelengthDivisions = 4;

layout(std430, binding = 1) buffer waves {
	float betaR[];
};

layout(std430, binding = 2) buffer xyz {
	vec4 xyzw[];
};

const int maxWavelengthDivisions = 48;
const float PI = 3.14159265358979323846;
const float earthRadius = 6360000;
const float atmosRadius = 6440000;

in vec2 texCoord;

out vec4 outputF;


vec3 intersectTopAtmosphere(vec3 origin, vec3 dir) {

	// project the center of the earth on to the ray
	vec3 u = vec3(-origin);
	// k is the signed distance from the origin to the projection
	float k = dot(dir,u);
	vec3 proj = origin + k * dir;
	
	// compute the distance from the projection to the atmosphere
	float aux = length(proj); 
	float dist = sqrt(atmosRadius * atmosRadius - aux*aux);
	
	dist += k;	
	return origin + dir * dist;
}


float distToTopAtmosphere(vec3 origin, vec3 dir) {

	// project the center of the earth on to the ray
	vec3 u = vec3(-origin);
	// k is the signed distance from the origin to the projection
	float k = dot(dir,u);
	vec3 proj = origin + k * dir;
	
	// compute the distance from the projection to the atmosphere
	float aux = length(proj); 
	float dist = sqrt(atmosRadius * atmosRadius - aux*aux);
	
	dist += k;	
	return dist;
}


void main()
{
	vec2 sunAnglesRad = vec2(sunAngles.x, sunAngles.y) * vec2(PI/180);
	vec3 sunDirection = vec3(cos(sunAnglesRad.y) * sin(sunAnglesRad.x),
							sin(sunAnglesRad.y),
							-cos(sunAnglesRad.y) * cos(sunAnglesRad.x));

	float angle = tan(fov * PI / 180.0 * 0.5);

	vec2 tc = texCoord * 2.0 - 1.0;
	vec2 pos = tc * vec2(ratio*angle, angle);
	vec3 dir = vec3(pos , -1);
	vec3 camRight = cross(camView, camUp);
	dir = camUp * pos.y + camRight * pos.x + camView;
	dir = normalize(dir);
	vec3 origin = vec3(0 ,earthRadius+1, 0);
	float dist = distToTopAtmosphere(origin, dir);
	
	float t0,t1;
	
	float rayleigh[maxWavelengthDivisions];
	for (int i = 0; i < wavelengthDivisions; ++i)
		rayleigh[i] = 0.0;
	float mie = 0.0;
	float cosViewSun = dot(dir, sunDirection);
	
	if (cosViewSun > 1) {
		outputF = vec4(1,0,0,0);
		return;
	}
	float opticalDepthRayleigh = 0;
	float opticalDepthMie = 0;

	// phase functions
	float phaseR = 0.75 * (1.0 + cosViewSun * cosViewSun);

	float aux = 1.0 + g*g - 2.0*g*cosViewSun;
	float phaseM = 3.0 * (1 - g*g) * (1 + cosViewSun * cosViewSun) / 
					(2.0 * (2 + g*g) * pow(aux, 1.5)); 

	float current = 0;
	float segLength = dist/divisions;
	
	for(int i = 0; i < divisions; ++i) {
		//segLength = current * quotient - current;
		vec3 samplePos = origin + (current + segLength * 0.5) * dir;
		float height = length(samplePos) - earthRadius;
		if (height < 0) {
			break;
			}
		float hr = exp(-height / Hr) * (segLength);
		float hm = exp(-height / Hm) * (segLength);
		opticalDepthRayleigh += hr;
		opticalDepthMie += hm;
		float distLightRay = distToTopAtmosphere(samplePos, sunDirection);
		float segLengthLight = distLightRay / divisonsLightRay;
		float currentLight = 0;
		float opticalDepthLightR = 0;
		float opticalDepthLightM = 0;
		int j = 0;
		for (; j < divisonsLightRay; ++j) {
			vec3 sampleLightPos = samplePos + (currentLight + segLengthLight * 0.5) * sunDirection;
			float heightLight = length(sampleLightPos) - earthRadius;
			if (heightLight < 0){
				break;
				}

			opticalDepthLightR += exp(-heightLight / Hr) * segLengthLight;
			opticalDepthLightM += exp(-heightLight / Hm) * segLengthLight;
			currentLight += segLengthLight;
		}
		if (j == divisonsLightRay) {
			float tau[maxWavelengthDivisions];
			float att[maxWavelengthDivisions];
			
			float aux = 4*PI * betaMf * (opticalDepthMie + opticalDepthLightM);
			for (int k = 0; k < wavelengthDivisions; ++k) {
				tau[k] = 4*PI * betaR[k] * (opticalDepthRayleigh + opticalDepthLightR) + aux;
				att[k] = exp(-tau[k]);
				rayleigh[k] += att[k] * hr;
				mie += att[k] * hm;// / wavelengthDivisions;
			}
		}
		current += segLength;
	}
	
	float result[maxWavelengthDivisions];
	for (int i = 0; i < wavelengthDivisions; ++i) {
		result[i] = rayleigh[i] * betaR[i] * phaseR + 0.9 * mie* betaMf * phaseM; 
	}
/*	if (cosViewSun >= 0.999192306417128873735516482698) {
		for (int i = 0; i < wavelengthDivisions; ++i) {
			result[i] =   rayleigh[i] /3000000;
		}
	}
*/	
	vec3 xyz = vec3(0);
	float w = 0;
	for (int i = 0; i < wavelengthDivisions; ++i) {
		xyz += result[i] * 
					xyzw[i].xyz * xyzw[i].w ;
		w += xyzw[i].w;
	}
	xyz = xyz/wavelengthDivisions ;
	//xyz /= wavelengthDivisions;
	
	mat3 XYZtoRGB3 = mat3(
		3.2404542,	-1.5371385,	-0.4985314,
		-0.969266,	1.8760108,	0.041556,
		0.0556434,	-0.2040259,	1.0572252
	);

	mat3 XYZtoRGB = mat3(
		2.6422874, -1.2234270, -0.3930143,
		-1.1119763,  2.0590183,  0.0159614,
		0.0821699, -0.2807254,  1.4559877
	);
	
	mat3 XYZtoRGB2 = mat3(
		0.41847, -0.15866, -0.082835,
		-0.091169, 0.25243, 0.015708,
		0.0009209, -0.0025498, 0.1786
	);
	
//	xyz += vec3(mie* betaMf * phaseM);
	vec3 res = transpose(XYZtoRGB3) * xyz ;
	res = res * 20;
	//res = vec3(0);// * 2;// * betaM[1][1] * phaseM)*20000000000000.00;
	//res = res/(1.0 + res);
	//res = clamp(res, vec3(0,0,0), vec3(100,100,100));
	//res = pow( clamp(smoothstep(0.0, 1.5	, log2(1.0+res)),0.0,1.0), vec3(1.0/2.2) );
	// tone mapping
	vec3 white_point = vec3(1-0);
	res = pow(vec3(1.0) - exp(-res / white_point * 1.5), vec3(1.0 / 2.2));
	
	outputF = vec4(res, 1);
	//outputF = vec4(result[0][3], result[2][1], result[3][2],1);

	

}


/*tonemapping operator
e.g. reinhard:
    atmos = pow( clamp(atmos / (atmos+1.0),0.0,1.0), vec3(1.0/2.2) );
or logarithmic:
    atmos = pow( clamp(smoothstep(0.0, 12.0, log2(1.0+atmos)),0.0,1.0), vec3(1.0/2.2) );
*/	