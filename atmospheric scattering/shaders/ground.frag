#version 430

uniform	sampler2D pos, normal;
uniform vec2 sunAngles;
uniform mat4 camView;
uniform vec3 camPos;
uniform int divisions = 4;
uniform int divisonsLightRay = 4;
uniform vec3 betaR = vec3(3.8e-6f, 13.5e-6f, 33.1e-6f);
uniform float betaMf = 21.0e-6f;
uniform float Hr = 7994;
uniform float Hm = 1200;
uniform float g = 0.99;

const float PI = 3.14159265358979323846;
const float earthRadius = 6360000;
const float atmosRadius = 6420000;

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

vec3 skyColor(vec3 dir, vec3 sunDir, vec3 origin, vec3 end) {

	vec3 betaM = vec3(betaMf);
	float distance = length(end - origin);
	float segLength = distance / divisions;
	vec3 rayleigh = vec3(0);
	vec3 mie = vec3(0);
	float cosViewSun = dot(dir, sunDir);
	if (cosViewSun > 1) {
		//outputF = vec4(0,1,0,0);
		return vec3(0,0,0);
	}
	float opticalDepthRayleigh = 0;
	float opticalDepthMie = 0;
//	float phaseR = 3.0 * (1.0 + cosViewSun * cosViewSun) / (16.0 * PI);
	float phaseR = (3/(16*PI)) * (1.0 + cosViewSun * cosViewSun) ;
	float aux = 1.0 + g*g - 2.0*g*cosViewSun;
	//if (aux < 0) aux = 0.0001; 
//	float phaseM = 1.5 * (1 - g*g) * (1 + cosViewSun * cosViewSun) / 
//					( (2 + g*g) * pow(aux, 1.5)); 
	float phaseM = 3.0 * (1 - g*g) * (1 + cosViewSun * cosViewSun) / 
					(8 * PI * (2 + g*g) * pow(aux, 1.5)); 
	//float phaseM = (1. - g*g)
	/// //---------------------------------------------
	//	((4. + PI) * pow(1. + g*g - 2.*g*cosViewSun, 1.5));
	float current = 0;
	for(int i = 0; i < divisions; ++i) {
		vec3 samplePos = origin + (current + segLength * 0.5) * dir;
		float height = length(samplePos) - earthRadius;
		if (height < 0) {
			return vec3(0.82,0.70,0.55);
			break;
			}
		float hr = exp(-height / Hr) * segLength;
		float hm = exp(-height / Hm) * segLength;
		opticalDepthRayleigh += hr;
		opticalDepthMie += hm;
		//raySphereIntersect(samplePos, sunDir, t0, t1);
		//vec3 light1 = samplePos + t1 * sunDir;
		vec3  light1 = intersectTopAtmosphere(samplePos, sunDir);
		float segLengthLight = length(light1 - samplePos) / divisonsLightRay;
		float currentLight = 0;
		float opticalDepthLightR = 0;
		float opticalDepthLightM = 0;
		int j = 0;
		for (; j < divisonsLightRay; ++j) {
			vec3 sampleLightPos = samplePos + (currentLight + segLengthLight * 0.5) * sunDir;
			float heightLight = length(sampleLightPos) - earthRadius;
			if (heightLight < 0){
				//outputF = vec4(1,1,0,0);
				//return;
				break;
				}

			opticalDepthLightR += exp(-heightLight / Hr) * segLengthLight;
			opticalDepthLightM += exp(-heightLight / Hm) * segLengthLight;
			currentLight += segLengthLight;
		}
		if (j == divisonsLightRay) {
			vec3 tau = betaR * (opticalDepthRayleigh + opticalDepthLightR) + 
					   betaM * 1.1 * (opticalDepthMie + opticalDepthLightM);
			vec3 att = exp(-tau);
			rayleigh += att * hr;
			mie += att * hm;
		}
		current += segLength;
	}
	vec3 result = (rayleigh * betaR * phaseR + mie * betaM * phaseM) * 20;
	result = clamp(result, vec3(0,0,0), vec3(100,100,100));
	//result = pow( clamp(result / (result+1.0),0.0,1.0), vec3(1.0/2.2) );
	result = pow( clamp(smoothstep(0.0, 1.5, log2(1.0+result)),0.0,1.0), vec3(1.0/2.2) );
	//result = vec3(1.0) – exp(-1.0 * result);
	/*result.x < 1.413? pow(result.x * 0.38317f, 1.0/2.2): 1.0 - exp(-result.x);
	result.y < 1.413? pow(result.y * 0.38317f, 1.0/2.2): 1.0 - exp(-result.y);
	result.z < 1.413? pow(result.z * 0.38317f, 1.0/2.2): 1.0 - exp(-result.z);
	*/
	return result;
}


vec3 computeAttenuation(float dist, float height) {

	vec3 betaM = vec3(betaMf);
	float hr = exp(-height / Hr) * dist;
	float hm = exp(-height / Hm) * dist;
	vec3 tau = betaR * hr + 
					   betaM * 1.1 * hm;
	vec3 att = exp(-tau);
	
	return att;
}

void main()
{
	vec4 position = texture(pos, texCoord);
	position.y += earthRadius;
	vec3 normal = texture(normal, texCoord).xyz;
	
	vec3 myPos = camPos + vec3(0,earthRadius, 0);
	
	if (normal == vec3(0,0,0)) {
		discard;
		return;
	}
		
	normal = normal * 2.0 - 1.0;	
	vec2 sunAnglesRad = vec2(sunAngles.x, sunAngles.y) * vec2(PI/180);
	vec3 sunDirection = vec3(cos(sunAnglesRad.y) * sin(sunAnglesRad.x),
							sin(sunAnglesRad.y),
							-cos(sunAnglesRad.y) * cos(sunAnglesRad.x));
	
	vec3 sunDirCam = vec3(camView * vec4(sunDirection, 0));
	
	vec3 objColor = computeAttenuation(length(position.xyz-myPos), myPos.y) * vec3(0.82,0.70,0.55) * max(dot(normal, sunDirCam), 0.25);
	if (dot(normal, sunDirCam) > 0 && sunAngles.y > 0.0) {
		// sun
		
		outputF = vec4(objColor + skyColor(normalize(position.xyz-myPos), sunDirection, myPos, position.xyz),1); 
	}
	else
		// shadow
		outputF = vec4(0.25 * objColor + skyColor(normalize(position.xyz-myPos), sunDirection, myPos, position.xyz),1);
}