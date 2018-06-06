#version 430

uniform vec3 camUp;
uniform vec3 camView;
uniform float ratio;
uniform float fov;

uniform vec2 sunAngles;
uniform float T;

#define FISH_EYE 1
#define REGULAR 0
uniform int cameraMode = REGULAR;
uniform float exposure = 1.5;

const float PI = 3.14159265358979323846;

in vec2 texCoord;

out vec4 outputF;

vec3 A,B,C,D,E;


vec3 F(float theta, float gamma) {

	return (1 + A * exp(B/cos(theta))) * (1 + C *exp(D*gamma) + E * pow(cos(gamma),2)); 
}


vec3 skyColor(vec3 dir, vec3 sunDir) {

	float cosTheta = dot(dir, vec3(0,1,0));
	float theta = acos(cosTheta);
	float cosThetaSun = dot(sunDir, vec3(0,1,0));
	float thetaSun = acos(cosThetaSun);
	float cosGamma = dot(sunDir, dir);
	float gamma = acos(cosGamma);
	
	vec2 TT = vec2(T,1);
	
	mat2x3 MA = mat2x3(
		0.1787,	-0.0193, -0.0167,
		-1.4630,-0.2592, -0.2608
	);
	mat2x3 MB = mat2x3(
		-0.3554, -0.0665, -0.0950,
		0.4275, 0.0008, 0.0092
	);
	mat2x3 MC = mat2x3(
		-0.0227, -0.0004, -0.0079,
		5.3251, 0.2125, 0.2102
	);	
	mat2x3 MD = mat2x3(
		0.1206, -0.0641, -0.0441,
		-2.5771, -0.8989, -1.6537
	);
	mat2x3 ME = mat2x3(
		-0.0670, -0.0033, -0.0109,
		0.3703, 0.0452, 0.0529
	);
	
	A = MA*TT; B = MB*TT; C = MC*TT; D = MD*TT; E = ME*TT;
	
	float tau = (4.0/9.0 - T/120) * (PI - 2 * thetaSun);
	float Y = (4.0453*T-4.9710) * tan(tau) - 0.2155*T + 2.4192;
	float tau0 = (4.0/9.0 - T/120) * (PI );
	float Y0 = (4.0453*T - 4.9710) * tan(tau0) - 0.2155*T + 2.4192;
	
	Y = Y/Y0;
	
	vec3 TTT = vec3(T*T, T, 1);
	
	mat4x3 Mx = mat4x3(
		 0.00166, -0.02903,  0.11693, 
		-0.00375,  0.06377, -0.21196,
		 0.00209, -0.03202,  0.06052,
		 0.00000,  0.00394,  0.25886
	);
	float x = dot(TTT,  (Mx * vec4(thetaSun*thetaSun*thetaSun, thetaSun*thetaSun, thetaSun, 1)));
	
	mat4x3 My = mat4x3(
		 0.00275, -0.04214,  0.15346, 
		-0.00610,  0.08970, -0.26756,
		 0.00317, -0.04153,  0.06670,
		 0.00000,  0.00516,  0.26688
	);
	float y = dot(TTT, My * vec4(thetaSun*thetaSun*thetaSun, thetaSun*thetaSun, thetaSun, 1));
	
	vec3 xyY = vec3(Y,x,y) * (F(theta, gamma) / F(0, thetaSun));
	
	xyY = xyY.yzx;
				
	vec3 XYZ = vec3(xyY.x * xyY.z / xyY.y, xyY.z, (1 -xyY.x - xyY.y)*xyY.z/xyY.y);
	
	mat3 XYZtoRGB = mat3(
		3.2404542,	-1.5371385,	-0.4985314,
		-0.969266,	1.8760108,	0.041556,
		0.0556434,	-0.2040259,	1.0572252
	);

	vec3 res = transpose(XYZtoRGB) * XYZ ;
	
	return res;
}


/*tonemapping operator
e.g. reinhard:
    atmos = pow( clamp(atmos / (atmos+1.0),0.0,1.0), vec3(1.0/2.2) );
or logarithmic:
    atmos = pow( clamp(smoothstep(0.0, 12.0, log2(1.0+atmos)),0.0,1.0), vec3(1.0/2.2) );
*/	

void main() {

	vec3 result;
	vec2 sunAnglesRad = vec2(sunAngles.x, sunAngles.y) * vec2(PI/180);
	vec3 sunDir = vec3(cos(sunAnglesRad.y) * sin(sunAnglesRad.x),
							 sin(sunAnglesRad.y),
							-cos(sunAnglesRad.y) * cos(sunAnglesRad.x));
							
	float angle = tan(fov * PI / 180.0 * 0.5);
	vec2 tc = texCoord * 2.0 - 1.0;
	
	if (cameraMode == REGULAR) { // normal camera
	
		vec2 pos = tc * vec2(ratio*angle, angle);
		vec3 dir = vec3(pos , -1);
		vec3 camRight = cross(camView, camUp);
		dir = camUp * pos.y + camRight * pos.x + camView;
		dir = normalize(dir);
		
		result = skyColor(dir, sunDir);
	}
	else { // fish eye camera

		float x = tc.x * ratio;
		float y = tc.y;
		float z = x*x + y*y;
		if (z < 1) {
			float phi = atan(y,x);
			float theta = acos(1-z);
			vec3 dir = vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
			
			result = skyColor(dir, sunDir);
		}
		else 
			result = vec3(0);
	}
		
	// tone mapping
	vec3 white_point = vec3(1.0);
	result = pow(vec3(1.0) - exp(-result / white_point * exposure), vec3(1.0 / 2.2));
	//result = pow( clamp(smoothstep(0.0, 2.0, log2(1.0+result)),0.0,1.0), vec3(1.0/2.2) );
	outputF = vec4(result, 1);
}