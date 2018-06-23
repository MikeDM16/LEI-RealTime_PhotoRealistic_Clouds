#version 430

uniform vec3 camUp = vec3(0,1,0);
uniform vec3 camView = vec3(0,0,-1);
uniform float fov = 65;
uniform float ratio = 1;

uniform int divisions = 32;
uniform int divisonsLightRay = 32;
uniform int cameraMode = 1;
uniform float exposure = 1.5;


uniform vec3 betaR = vec3(3.8e-6f, 13.5e-6f, 33.1e-6f);
uniform float betaMf = 2.1e-5f;
uniform float Hr = 7994;
uniform float Hm = 1200;
uniform float g = 0.99;
uniform vec2 sunAngles = vec2(28,28);

const float PI = 3.14159265358979323846;
const float earthRadius = 6360000;
const float atmosRadius = 6420000;
const float fourPI = 4.0 * PI;

in vec2 texCoord;

out vec4 outputF;

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


vec3 skyColor(vec3 dir, vec3 sunDir, vec3 origin, float dist) {

	float cosViewSun = dot(dir, sunDir);
	
	vec3 betaM = vec3(betaMf);
	
	vec3 rayleigh = vec3(0);
	vec3 mie = vec3(0);
	
	float opticalDepthRayleigh = 0;
	float opticalDepthMie = 0;

	// phase functions
	float phaseR = 0.75 * (1.0 + cosViewSun * cosViewSun);

	float aux = 1.0 + g*g - 2.0*g*cosViewSun;
	float phaseM = 3.0 * (1 - g*g) * (1 + cosViewSun * cosViewSun) / 
					(2.0 * (2 + g*g) * pow(aux, 1.5)); 

	float segLength = dist / divisions;
	float current = 0;
	float height;
	for(int i = 0; i < divisions; ++i) {
		vec3 samplePos = origin + (current + segLength * 0.5) * dir;
		height = length(samplePos) - earthRadius;
		if (height < 0) {
			break;
		}
		float hr = exp(-height / Hr) * segLength;
		float hm = exp(-height / Hm) * segLength;
		opticalDepthRayleigh += hr;
		opticalDepthMie += hm;
		
		float distLightRay = distToTopAtmosphere(samplePos, sunDir);
		float segLengthLight = distLightRay / divisonsLightRay;
		float currentLight = 0;
		float opticalDepthLightR = 0;
		float opticalDepthLightM = 0;
		int j = 0;
		for (; j < divisonsLightRay; ++j) {
			vec3 sampleLightPos = samplePos + (currentLight + segLengthLight * 0.5) * sunDir;
			float heightLight = length(sampleLightPos) - earthRadius;
			if (heightLight < 0){
				break;
			}

			opticalDepthLightR += exp(-heightLight / Hr) * segLengthLight;
			opticalDepthLightM += exp(-heightLight / Hm) * segLengthLight;
			currentLight += segLengthLight;
		}
		if (j == divisonsLightRay) {
			vec3 tau = fourPI * betaR * (opticalDepthRayleigh + opticalDepthLightR) + 
					   fourPI * betaM *  (opticalDepthMie + opticalDepthLightM);
			vec3 att = exp(-tau);
			rayleigh += att * hr;
			mie += att * hm;
		}
		current += segLength;
	}
	vec3 result = (rayleigh *betaR * phaseR + mie * 0.9 *betaM * phaseM) * 20;

	/*	if (cosViewSun >= 0.999192306417128873735516482698) {
		result =   exp(-opticalDepthRayleigh * betaR);// - opticalDepthMie * betaM) ;
		//result *=   exp(- opticalDepthMie * betaM) ;
		
	}
*/
	return result;
}


void main() {

	vec3 result;
	vec2 sunAnglesRad = vec2(sunAngles.x, sunAngles.y) * vec2(PI/180);
	vec3 sunDir = vec3(cos(sunAnglesRad.y) * sin(sunAnglesRad.x),
							 sin(sunAnglesRad.y),
							-cos(sunAnglesRad.y) * cos(sunAnglesRad.x));
	float angle = tan(fov * PI / 180.0 * 0.5);
	vec3 origin = vec3(0.0, earthRadius+1, 0.0);
	vec2 tc = texCoord * 2.0 - 1.0;
	
	if (cameraMode == 1) { // normal camera
	
		vec2 pos = tc * vec2(ratio*angle, angle);
		vec3 dir = vec3(pos , -1);
		vec3 camRight = cross(camView, camUp);
		dir = camUp * pos.y + camRight * pos.x + camView;
		dir = normalize(dir);
		float dist = distToTopAtmosphere(origin, dir);
		
		result = skyColor(dir, sunDir, origin, dist);
	}
	else { // fish eye camera

		float x = tc.x * ratio;
		float y = tc.y;
		float z = x*x + y*y;
		if (z < 1) {
			float phi = atan(y,x);
			float theta = acos(1-z);
			vec3 dir = vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
			float dist = distToTopAtmosphere(origin, dir);
			
			result = skyColor(dir, sunDir, origin, dist);
		}
		else 
			result = vec3(0);
	}
	// tone mapping
	vec3 white_point = vec3(1.0);
	result = pow(vec3(1.0) - exp(-result / white_point * exposure), vec3(1.0 / 2.2));
	
	outputF = vec4(result, 1);

}