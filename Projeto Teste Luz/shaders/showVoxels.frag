#version 440

out vec4 FragColor;

uniform vec4 lDiffuse;
uniform vec4 lPosition;
uniform vec4 lDir; 
uniform	mat4 m_view;

/*in Data {
	vec3 l_dir;
} DataIn;
*/

uniform sampler2D shapeNoise;
uniform int shapeWidth;
uniform int shapeHeight;

uniform sampler2D erosionNoise;
uniform int erosionWidth;
uniform int erosionHeight;

uniform sampler2D weatherTexture;
uniform int weatherWidth;
uniform int weatherHeight;

uniform int GridSize;

uniform mat4 VM;
uniform float FOV;
uniform float RATIO;
uniform vec2 WindowSize;
uniform vec3 RayOrigin;
uniform int level = 0;
uniform vec3 aabbMin, aabbMax;

// Parameters from inteface
uniform float layer_Height;
uniform float g_phase_function;
uniform float sigmaExtintion;
uniform float sigmaAbsorption;
uniform float sigmaScattering;

// needed for the sky color
 
uniform float betaMf = 21.0e-6f;
const int maxWavelengthDivisions = 48;
uniform int wavelengthDivisions = 4;
uniform float Hr = 7994;
uniform int divisions = 32;
uniform int divisionsLightRay = 32;
uniform float exposure = 1.5;
uniform float Hm = 1200;
const float PI = 3.14159265358979323846;
const float earthRadius = 6360000;
const float atmosRadius = 6440000;
layout(std430, binding = 1) buffer waves {
	float betaR[];
};

layout(std430, binding = 2) buffer xyz {
	vec4 xyzw[];
};

struct Ray {
    vec3 Origin;
    vec3 Dir;
};

struct AABB {
    vec3 Min;
    vec3 Max;
};

float HeightSignal(vec3 pos, float h_start, float h_cloud);
float getShape(vec3 pos);
float getErosion(vec3 pos);
float getDensity(vec3 pos);
float Transmittance(float density, float l, vec3 step_dir);

bool IntersectBox(Ray r, out float t0, out float t1)
{
    vec3 invR = 1.0 / r.Dir;
    vec3 tbot = invR * (aabbMin-r.Origin);
    vec3 ttop = invR * (aabbMax-r.Origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    // vec2 t = max(tmin.xx, tmin.yz);
    // t0 = max(t.x, t.y);
    // t = min(tmax.xx, tmax.yz);
    // t1 = min(t.x, t.y);
 	 t0 = max(tmin.x, max(tmin.y, tmin.z));
	 t1 = min(tmax.x, min(tmax.y, tmax.z));
   return t0 <= t1;
}


//------------------------------------------------------------------------
float HeightSignal(vec3 pos, float h_start, float h_cloud) {
    // h_start = altitude onde inicia a nuvem
    // h_cloud = altura da nuvem, conforme o volume da mesma
    // h_start e h_cloud estão entre [0..1]
    // pos --> retirar a altura onde está o pos na fase a Ray Marching

    float atm = pos.y * layer_Height;
    float r  = (atm - h_start)*(atm - h_start - h_cloud);
    r *= (-4 / (h_cloud * h_cloud + 0.00001));
    return r;
}
//------------------------------------------------------------------------

// -------- Funções de Shape -> gives general form to the cloud  ---------- */
float getShape(vec3 pos){
    // Normalize the coords for the 3D noise size
    vec3 aux = vec3(pos);
    aux.xyz *= shapeHeight;
    aux.x = floor(aux.y)*(shapeHeight/4) + aux.x;
    vec2 textCoord = aux.xz;

    float densidadeR = texelFetch(shapeNoise, ivec2(textCoord), level).r;
    return densidadeR;
    
    //float densidadeG = texelFetch(shapeNoise, ivec2(textCoord), level).g;
    //float densidadeB = texelFetch(shapeNoise, ivec2(textCoord), level).b;
    //float densidadeA = texelFetch(shapeNoise, ivec2(textCoord), level).a;
    //return 0.625*densidadeG * 0.25*densidadeB * 0.125*densidadeA; ;
}
//------------------------------------------------------------------------

// -------- Funções de Erosion -> gives details to cloud  ---------------- */
float getErosion(vec3 pos){
    // Normalize the coords for the 3D noise size
    vec3 aux = vec3(pos);
    aux.xyz *= erosionHeight;
    aux.x = floor(aux.y)*erosionHeight + aux.x;
    vec2 textCoord = aux.xz;
    
    float densidadeR =  texture(erosionNoise, vec2(textCoord.x/erosionWidth, pos.z), level).r;
    float densidadeG =  texture(erosionNoise, vec2(textCoord.x/erosionWidth, pos.z), level).g;
    float densidadeB =  texture(erosionNoise, vec2(textCoord.x/erosionWidth, pos.z), level).b;
    
    return  0.625*densidadeR * 0.25*densidadeG * 0.125*densidadeB; 
}
//------------------------------------------------------------------------

// -------------------- Funções de heightGradiente ------------------------ */
float density_gradient_stratus(const float h){
    return max(smoothstep(0.00, 0.07, h) - smoothstep(0.07, 0.11, h), 0); // stratus, could be better
}

float density_gradient_cumulus(const float h){
    return max(smoothstep(0.00, 0.22, h) - smoothstep(0.4, 0.62, h), 0); // cumulus
    //return smoothstep(0.3, 0.35, h) - smoothstep(0.425, 0.7, h); // cumulus
}

float density_gradient_cumulonimbus(const float h){
    return max(smoothstep(0.0, 1.0, h) - smoothstep(0.9, 1.0, h), 0); // cumulonimbus
}

float HeightGradient(vec3 pos, float h_start, float h_cloud) {
    float atenuacao; 
    // Nuvens rasteiras e com pouca altura
    if((h_start < 0.1) && (h_cloud < 0.3) ){
        atenuacao = density_gradient_stratus(pos.y);
    }else
        if((h_start < 0.5) && (h_cloud < 0.6)){
            atenuacao = density_gradient_cumulus(pos.y);
            //atenuacao = texelFetch(shapeNoise, ivec2(textCoord), level).a;
        }else
            atenuacao = density_gradient_cumulonimbus(pos.y);

    return atenuacao * pos.y;
}
//------------------------------------------------------------------------

/* ----- Função para determinar a densidade numa determinada posição ----- */
float getDensity(vec3 pos){
    vec3 aux = vec3(pos);
    aux.x *= weatherHeight;
    aux.z *= weatherWidth;
    //aux.x = floor(aux.y)*(weatherHeight) + aux.x;
    vec2 textCoord = aux.xz;

    // Densidade inicial obtida da weather texture
    vec3 weather = texelFetch(weatherTexture, ivec2(textCoord), level).rgb;
    float density = weather.r;

    // Aplicação da função Height signal
    density *= HeightSignal(pos, weather.b, weather.g);

    //--- Fase da Shape  ---
    density *= getShape(pos);

    //--- Fase da Erosion ---
    density -= getErosion(pos);

    // Only use positive densitys after erosion !
    if(density > 0){
        density *= HeightGradient(pos, weather.b, weather.g);

        // clamp density to 1 for more balance lightning
        if(density > 1)
            density = 1;
    }

    return density;
}
//------------------------------------------------------------------------
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


vec4 skyColor(vec3 dir, vec3 sunDir, vec3 origin, float g) {
    float dist = distToTopAtmosphere(origin, sunDir);

	float rayleigh[maxWavelengthDivisions];
	for (int i = 0; i < wavelengthDivisions; ++i)
		rayleigh[i] = 0.0;
	float mie = 0.0;
	float cosViewSun = dot(dir, sunDir);
	
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
	
		vec3 samplePos = origin + (current + segLength * 0.5) * dir;
		float height = length(samplePos) - earthRadius;
		
		if (height < 0) {
			break;
		}
		
		float hr = exp(-height / Hr) * (segLength);
		float hm = exp(-height / Hm) * (segLength);
		opticalDepthRayleigh += hr;
		opticalDepthMie += hm;
		
		float distLightRay = distToTopAtmosphere(samplePos, sunDir);
		float segLengthLight = distLightRay / divisionsLightRay;
		float currentLight = 0;
		
		float opticalDepthLightR = 0;
		float opticalDepthLightM = 0;
		
		int j = 0;
		for (; j < divisionsLightRay; ++j) {
			vec3 sampleLightPos = samplePos + (currentLight + segLengthLight * 0.5) * sunDir;
			float heightLight = length(sampleLightPos) - earthRadius;
			
			if (heightLight < 0){
				break;
				}

			opticalDepthLightR += exp(-heightLight / Hr) * segLengthLight;
			opticalDepthLightM += exp(-heightLight / Hm) * segLengthLight;
			currentLight += segLengthLight;
		}
		if (j == divisionsLightRay) {
			float tau[maxWavelengthDivisions];
			float att[maxWavelengthDivisions];
			
/*			float aux = 4*PI * betaMf * (opticalDepthMie + opticalDepthLightM);
			for (int k = 0; k < wavelengthDivisions; ++k) {
				tau[k] = 4*PI * betaR[k] * (opticalDepthRayleigh + opticalDepthLightR) ;
				att[k] = exp(-tau[k]);
				rayleigh[k] += att[k] * hr;
				att[k] = exp(-aux);
				mie += att[k] * hm / wavelengthDivisions;
			}
*/			float aux = 4*PI * 1.1 * betaMf * (opticalDepthMie + opticalDepthLightM);
			for (int k = 0; k < wavelengthDivisions; ++k) {
				tau[k] = 4*PI * betaR[k] * (opticalDepthRayleigh + opticalDepthLightR) + aux;
				att[k] = exp(-tau[k]);
				rayleigh[k] += att[k] * hr;
				mie += att[k] * hm / wavelengthDivisions;
			}
		}
		current += segLength;
	}
	
	float result[maxWavelengthDivisions];
	for (int i = 0; i < wavelengthDivisions; ++i) {
		result[i] = rayleigh[i] * betaR[i] * phaseR + mie* betaMf * phaseM; 
	}
	
	/*if (cosViewSun >= 0.999192306417128873735516482698) {
		for (int i = 0; i < wavelengthDivisions; ++i) {
			result[i] =   exp(-4*PI*opticalDepthRayleigh * betaR[i] -4*PI * betaMf * opticalDepthMie)*0.05 ;
		}
	}*/
	
	vec3 xyz = vec3(0);
	float w = 0;
	for (int i = 0; i < wavelengthDivisions; ++i) {
		xyz += result[i] * xyzw[i].xyz * xyzw[i].w ;
		w += xyzw[i].w;
	}
	xyz = xyz/wavelengthDivisions ;
	//xyz /= wavelengthDivisions;
	
    // tone mapping
    
	vec3 white_point = vec3(1.0);
	vec3 res = xyz; 
    res = pow(vec3(1.0) - exp(-res / white_point * exposure), vec3(1.0 / 2.2));
	
    return vec4(res,1);

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
	
	res = transpose(XYZtoRGB3) * xyz ;
	//res = res * 20;
	
	return vec4(res,1);
}

//------------------------------------------------------------------------
/* Determinate the direct light coming from the light source and possibly ocluded 
from other clouds 
   - Does another Ray Marching through the volume, but from the step_pos to the sun position
   - Evalute the step_position density in every step of Ray Marching
   - Ultra-Mega fps killer */
vec4 computDirectLight(vec3 step_pos){
    // Create an Ray from the step position to the light source 
    //vec3 dir_to_sun = vec3(lPosition) - step_pos; 
    vec3 ldir_n = normalize(vec3(m_view * lDir));
    Ray ray_to_sun = Ray( step_pos, normalize(ldir_n) );

    // Same code to adjust the iterations to the box size 
    float tnear, tfar;
    bool r = IntersectBox(ray_to_sun, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;

    vec3 rayStart = ray_to_sun.Origin + ray_to_sun.Dir * tnear;
    vec3 rayStop = ray_to_sun.Origin + ray_to_sun.Dir * tfar;

    vec3 len = aabbMax - aabbMin;
    rayStart = 1/len * (rayStart -aabbMin);
    rayStop = 1/len * (rayStop -aabbMin);

    int steps = 25;
    vec3 step = (rayStop-rayStart) / float(steps);
    vec3 pos = rayStart + 0.5 * step;
    int travel = steps;

    // compute the possible oclusion of other clouds to the direct sun light 
    vec4 color = vec4(0.0);
    vec4 l_sun = vec4(lDiffuse.rgb, 1); 

    for (travel = 0; travel != steps; travel++) {
        float density = getDensity(pos);

        if(density > 0 ){
            /*---   Transmittance    ---*/
            float l = length(rayStart - pos); // distance the light will travel through the volume
            float transmittance = Transmittance(density, l, step); 
            
            color *=  l_sun * transmittance;
        }

        /*Marching towards the sun has a great impact on performance since for 
        every step an extra number of steps has to be taken. 
        In our implementation four steps towards the sun are taken at 
        exponentially increasing steps size (pág 29 tese)*/
        pos += step; //pow(4, travel) * step;
    }

    return (color); 
}
//------------------------------------------------------------------------


// -------------------- Funções de Fase para scattering ------------------ */
/*  Henyey-Greenstein function (Mie Phase Function):
        light : direção da luz
        step_dir : "direção da camara" no percurso do Ray Marching
    f(x) = (1 - g^2) / (4PI * (1 + g^2 - 2g*cos(teta))^[3/2])        */
float phase_functionHG(float g , vec3 light, vec3 step_dir) {

    // 1 - g^2
	float n = 1 - pow(g , 2); 
	
    // 1 + g^2 - 2g*cos(x)
    float cos_teta = dot(light,step_dir); // cos(x)
	float d = 1 + pow(g ,2) - 2 * g * cos_teta; 
    
    return  float(n  / (4*PI * pow(d, 1.5)));
}

/*  Cornette-Shank aproach
    This phase function is also well suited for clouds but is more 
 time consuming to calculate
    f(x) = 3*(1 - g^2) *       (1 + cos^2(teta))
           2*(2 + g^2)   (1 + g^2 -2g*cos(teta^[3/2]))  */
float phase_functionCS(float g, vec3 light, vec3 step_dir) {
	// 3*(1 - g^2) / 2*(2 + g^2)
	float n = (3/2) * (1 - pow(g, 2))/(2+pow(g, 2)); 
	
    // (1 + cos^2(teta)) / (1 + g^2 -2g*cos(teta^[3/2]))
    float cos_teta = dot(light,step_dir); // cos(x)
	float d = 1 + pow(g, 2) - 2*g*cos_teta; 
    return n * (1 + pow(cos_teta, 2)) / pow(d, 1.5);
}

/*  The g parameter varies from −1 ≤ g ≤ 1
        - Backscattering:  −1 ≤ g < 0 
        - Isotropic scatterin: g = 0
        - Forward scattering: 0 < g ≤ 1
*/
vec4 Scattering(float g, vec3 step_pos, vec3 step_dir){
    vec3 dir_to_sun = normalize(vec3(m_view * lDir));
 
    // Henyey-Greenstein function:
    float phase =  10*phase_functionHG(g, dir_to_sun, step_dir);
   
    // or Cornette-Shank aproach
    // float phase =  phase_functionCS(g, dir_to_sun, step_dir);

    //vec4 sun_light = computDirectLight(step_pos);
    vec4 ambiente_light = vec4(0.2,0.2,0.2,1.0); //skyColor(step_dir, dir_to_sun, step_pos, g);
    vec4 sun_light = vec4(1); 

    return (sun_light * phase) + ambiente_light;

}
//------------------------------------------------------------------------

//------------------------------------------------------------------------
/*   Transmittance Tr is the amount of photos that travels unobstructed between
 two points along a straight line. The transmittance can be calculated using 
 Beer-Lambert’s law     
    T = e^( -(sigma_Abs+sigma_Ext) * l ) 
    sigma_Abs - sigma absorption [0,inf] 
    sigma_Ext - sigma extintion  [0,inf]
    l - distance between the atual position and star of the box 
        ~ distance travel by the light in the box, going to the camera */
float Transmittance(float density, float l, vec3 step_dir){
    float sigmaAbs = sigmaAbsorption * density; // sigma absorption
    float sigmaExt = sigmaExtintion  * density; // sigma Extintion 
    float sigmaScatt = sigmaScattering * density; // sigma Scattering 
    
    // Beers Law  E = exp(-l)
    //return exp(- l * density);

    // Powder Law E = 1 - exp(-l*2)
    //return  (1 - exp(- l * 2 ));

    // Beer's Powder 
    //return exp(-(sigmaAbs + sigmaExt) * l);

    // proposta de Juraj Palenik master thesis 
    vec3 dir_to_sun = normalize(vec3(m_view * lDir));
    float cos_teta = dot(dir_to_sun, step_dir); // cos(x)
    float BP_exp1 = exp(-(sigmaScatt)*l);
    float BP_exp2 = ((cos_teta + 1)/2) * exp(-(sigmaAbs + sigmaExt) * l); 
    return  BP_exp1 * BP_exp2;  
}

vec4 ComputLight(vec3 RayOrigin, vec3 rayDirection, vec3 atual_pos, float density){ 
    vec4 final_color = vec4(1.0);
    float Tr = 1; 

    /*---   Transmittance    ---*/
    float l = length(RayOrigin - atual_pos); // distance the light will travel through the volume
    float transmittance = Transmittance(density, l, rayDirection); 
    
    /*---   Scattering    ---*/
    float g = g_phase_function * density; // sigma scattering   
    vec4 scatter_light = Scattering(g, atual_pos, rayDirection);

    /*---   Light Influence  (ciclo até ao sol)  ---*/
    vec4 intensity =  computDirectLight(atual_pos);
    
    /*---   Combine everything   ---*/
    vec3 dir_to_sun = normalize(vec3(m_view * lDir));
    
    //vec4 ambiente_light = skyColor(rayDirection, dir_to_sun, RayOrigin, density);
    
    return transmittance * scatter_light + (intensity); 

}

void main() {

    float FocalLength = 1.0/ tan(radians(FOV*0.5));
    vec3 rayDirection;
    rayDirection.xy = 2.0 * gl_FragCoord.xy / WindowSize.xy - 1.0;
    rayDirection.xy *= vec2(RATIO,1);
    rayDirection.z = -FocalLength;
    rayDirection = (vec4(rayDirection, 0) * VM).xyz;

    Ray eye = Ray( RayOrigin, normalize(rayDirection) );

    float tnear, tfar;
    bool r = IntersectBox(eye, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;

    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;

    vec3 len = aabbMax - aabbMin;
    rayStart = 1/len * (rayStart -aabbMin);
    rayStop = 1/len * (rayStop -aabbMin);

    double larg = aabbMax.x - aabbMin.x;
    double cump = aabbMax.z - aabbMin.z;
    double alt  = aabbMax.y - aabbMin.y;

    int steps = int(0.5 + distance(rayStop, rayStart)  * float(GridSize) * 2);
    vec3 step = (rayStop-rayStart) / float(steps);
    vec3 pos = rayStart + 0.5 * step;
    int travel = steps;
    
    vec4 color =  vec4(0.0);
    float Tr = 1; 
    for (;  /*color.w == 0  && */ travel != 0;  travel--) {
        
        float density = getDensity(pos);
        
        if(density > 0){
            color += ComputLight(rayStart, rayDirection, pos, density);
            //color += 0.02 * clamp(ComputLight(rayStart, rayDirection, pos, density), 0.0, 1.0);
        }

        pos += step;
    }

    // tonemapping operator
    // Reinhard: pow( clamp(atmos / (atmos+1.0),0.0,1.0), vec3(1.0/2.2) );
    //color = pow( clamp(color/(color + 1.0), 0.0, 1.0), vec4(15.0/2.2) );
    
    // Logarithmic: pow( clamp(smoothstep(0.0, 12.0, log2(1.0+atmos)),0.0,1.0), vec3(1.0/2.2) ); */
    //color = pow( clamp(smoothstep(0.0, 8.0, log2(1.0+color)),0.0,1.0), vec4(1.0/2.2) );

    // tone mapping
	vec4 white_point = vec4(1.0);
	color = pow(vec4(1.0) - exp(-color / white_point * 0.02), vec4(1.0 / 2.2));

    FragColor = color;
}
