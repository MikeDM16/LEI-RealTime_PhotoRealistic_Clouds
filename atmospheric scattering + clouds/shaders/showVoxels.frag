#version 440


in Data {
	vec3 l_dir;
    vec2 texCoord;
} Datain;

out vec4 FragColor;
/*------------------------------------------------------------------------
------------------------------------------------------------------------*/
uniform vec4 lDiffuse;
uniform vec4 lPosition;
uniform vec4 lDir; 
uniform	mat4 m_view;

uniform sampler2D shapeNoise;
uniform int shapeWidth;
uniform int shapeHeight;

uniform sampler2D erosionNoise;
uniform int erosionWidth;
uniform int erosionHeight;
uniform float threshold_erosion;
uniform float erosion_amount;

uniform sampler2D weatherTexture;
uniform int weatherWidth;
uniform int weatherHeight;

uniform mat4 VM;
uniform float FOV;
uniform float RATIO;
uniform vec2 WindowSize;
uniform vec3 RayOrigin;
uniform int level = 0;
uniform vec3 aabbMin, aabbMax;

// Parameters from inteface
uniform float layer_Height;
uniform float g0_phase_function;
uniform float g1_phase_function;
uniform float phase_mix;
uniform float sigmaAbsorption;
uniform float sigmaScattering;
uniform float k_transmittance; 
uniform int volume_steps; // for Ray Marching 
uniform float gamma; // for tone mapping 
/*------------------------------------------------------------------------
------------------------------------------------------------------------*/
uniform vec3 camUp = vec3(0,1,0);
uniform vec3 camView = vec3(0,0,-1);
uniform float fov = 65;
uniform float ratio = 1;

uniform int divisions = 32;
uniform int divisionsLightRay = 32;
uniform float exposure = 1.5;

#define FISH_EYE 1
#define REGULAR 0
uniform int cameraMode = REGULAR;

#define LINEAR 0
#define EXPONENTIAL 1
uniform int sampling = LINEAR;

uniform vec3 betaR = vec3(3.8e-6f, 13.5e-6f, 33.1e-6f);
uniform float betaMf = 2.1e-5f;
uniform float Hr = 7994;
uniform float Hm = 1200;
uniform float g = 0.99;
uniform vec2 sunAngles;

const float PI = 3.14159265358979323846;
const float earthRadius = 6360000;
const float atmosRadius = 6420000;
const float fourPI = 4.0 * PI;
/*------------------------------------------------------------------------
------------------------------------------------------------------------*/
struct Ray {
    vec3 Origin;
    vec3 Dir;
};

struct AABB {
    vec3 Min;
    vec3 Max;
};
/*------------------------------------------------------------------------
------------------------------------------------------------------------*/
float HeightSignal(vec3 pos, float h_start, float h_cloud);
float getShape(vec3 pos);
float getErosion(vec3 pos);
float getDensity(vec3 pos);
float Transmittance(float density, float l, vec3 step_dir, vec3 dir_to_sun);
/*------------------------------------------------------------------------
------------------------------------------------------------------------*/
float distToTopAtmosphere(vec3 origin, vec3 dir);
void initSampling(in float dist, in int div, out float quotient, out float segLength);
void computeSegLength(float quotient, float current, inout float segLength);
vec3 skyColor(vec3 pos);
/*------------------------------------------------------------------------
------------------------------------------------------------------------*/
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

    float sky_start = 0; // review this 

    float atm = sky_start + pos.y *layer_Height;
    //h_start   = sky_start + h_start*layer_Height;
    //h_cloud   = h_cloud * layer_Height; 

    float r  = (atm - h_start)*(atm - h_start - h_cloud);
    r *= (-4 / (h_cloud * h_cloud + 0.00001));
    return r;

}
//------------------------------------------------------------------------

// -------- Funções de Shape -> gives general form to the cloud  ---------- */
float getShape(vec3 pos){
    // Normalize the coords for the 3D noise size
    vec3 aux = vec3(pos);
    aux.xz *= shapeHeight;
    aux.y *= 32; 
    aux.x = floor(aux.y)*(shapeHeight) + aux.x;
    vec2 textCoord = aux.xz;

    // Clean some of the bad residual clouds in the margins of the volume
    if((pos.z < 0.05 || pos.z > 0.95) || (pos.x < 0.05 || pos.x > 0.95)) {
        return 0.0;
    }

    vec4 densidades = texelFetch(shapeNoise, ivec2(textCoord), 0).rgba;

    float densidadeR = densidades.r;
    float densidadeG = densidades.g;
    float densidadeB = densidades.b;
    float densidadeA = densidades.a;

    return 0.625*densidadeR + 0.5*densidadeG + 0.25*densidadeB + 0.125 * densidadeA; 
    
    //return 0.625*densidadeR * 0.5*densidadeG * 0.125*densidadeB; 
    //return densidadeR* densidadeG *densidadeB * densidadeA ;
}
//------------------------------------------------------------------------

// -------- Funções de Erosion -> gives details to cloud  ---------------- */
float getErosion(vec3 pos){
    // Normalize the coords for the 3D noise size
    vec3 aux = vec3(pos);
    aux.xyz *= erosionHeight;
    aux.x = floor(aux.y)*erosionHeight + aux.x;
    vec2 textCoord = aux.xz;
    
    vec3 densidades = texelFetch(erosionNoise, ivec2(textCoord), 0).rgb; 
    //vec3 densidades = texture(erosionNoise, ivec2(textCoord.x/erosionWidth, pos.z), level).rgb; 

    float densidadeR =  densidades.r;
    float densidadeG =  densidades.g;
    float densidadeB =  densidades.b;
    
    return densidadeR * densidadeG * densidadeB; 
    //return 0.625*densidadeR * 0.5*densidadeG * 0.125*densidadeB; 
}
//------------------------------------------------------------------------

// -------------------- Funções de heightGradiente -----------------------
float density_gradient_stratus(const float h){
    return smoothstep(0.00, 0.26, h);
    //return max(smoothstep(0.0, 0.05, h) - smoothstep(0.05, 0.22, h), 0.0); // stratus, could be better
}

float density_gradient_cumulus(const float h){
    return smoothstep(0.25, 0.5, h) - smoothstep(0.45, 0.7, h); // cumulus
}

float density_gradient_cumulonimbus(const float h){
    return smoothstep(0.16, 1.0, h) - smoothstep(0.65, 1.0, h); //, 0); // cumulonimbus
}

float HeightGradient(vec3 pos, float cloudType, float h_cloud) {
    float atenuacao = 1.0; 
    /*
    const vec4 STRATUS_GRADIENT = vec4(0.0f, 0.09f, 0.015, 0.20);
    const vec4 STRATOCUMULUS_GRADIENT = vec4(0.1f, 0.2f, 0.48f, 0.5f);
    const vec4 CUMULUS_GRADIENT = vec4(0.01f, 0.0625f, 0.058f, 1.0f); 
    // these fractions would need to be altered if cumulonimbus are added to the same pass
    
    vec3 pesos; 
    if( cloudType < 0.1)
        pesos = vec3(1.0, 1.10, 0.0);
    else if(cloudType < 0.6)
        pesos = vec3(0.10, 1.0, 0.20);
    else 
        pesos = vec3(0.0, 0.0, 1.0);
    
    
    // Mix gradients 
    float stratus = 1.0f - clamp(floor(cloudType) * 2.0f, 0.0, 1.0);
    float stratocumulus = 1.0f - abs(cloudType - 0.5f) * 2.0f;
    float cumulus = clamp(cloudType - 0.5f, 0.0, 1.0) * 2.0f;

    vec4 GradientPesos = STRATUS_GRADIENT * pesos.x + STRATOCUMULUS_GRADIENT * pesos.y + CUMULUS_GRADIENT * pesos.z;
    float grad_pesos = smoothstep(GradientPesos.x, GradientPesos.y, pos.y) 
                            - smoothstep(GradientPesos.z, GradientPesos.w, pos.y);

    vec4 cloudGradient = STRATUS_GRADIENT * stratus + STRATOCUMULUS_GRADIENT * stratocumulus + CUMULUS_GRADIENT * cumulus;
    float grad_cloud = smoothstep(cloudGradient.x, cloudGradient.y, pos.y) 
                            - smoothstep(cloudGradient.z, cloudGradient.w, pos.y);
    
    return grad_cloud;
    return mix(grad_cloud, grad_pesos,  0.5);
    */

    // Nubens baixas e com pouca altitude - Stratus
    if( (cloudType < 0.15)){
        atenuacao = density_gradient_stratus(pos.y);
    }else
        // nuvens que se encontrem a meia altitude da atmosfera - cumulus 
        if((cloudType < 0.6) ){
            atenuacao = density_gradient_cumulus(pos.y);
        }else{
            atenuacao = density_gradient_cumulonimbus(pos.y);
            
        }

    return atenuacao;
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
    float density = weather.r; // cloud coverage

    // Aplicação da função Height signal
    density *= HeightSignal(pos, weather.b, weather.g);

    //--- Fase da Shape  ---
    density *= getShape(pos);

    //--- Fase da Erosion ---
 
    if(density < threshold_erosion)
        density -= erosion_amount *  getErosion(pos);
    
    // Only use positive densitys after erosion !
    if(density > 0.0)
       density *= HeightGradient(pos, weather.b, weather.g);
    
    // clamp density to 1 for more balance lightning
    density = clamp(density, 0.0, 1.0);

    return density;
}
//------------------------------------------------------------------------

//------------------------------------------------------------------------
/* Determinate the direct light coming from the light source and possibly ocluded 
from other clouds 
   - Does another Ray Marching through the volume, but from the step_pos to the sun position
   - Evalute the step_position density in every step of Ray Marching
   - Ultra-Mega fps killer */
vec3 computDirectLight(vec3 step_pos, vec3 dir_to_sun){

    Ray ray_to_sun = Ray( step_pos, dir_to_sun );

    // Same code to adjust the iterations to the box size 
    float tnear, tfar;
    bool r = IntersectBox(ray_to_sun, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;

    vec3 rayStart = ray_to_sun.Origin;
    vec3 rayStop = ray_to_sun.Origin + ray_to_sun.Dir * tfar;

    vec3 len = aabbMax - aabbMin;
    rayStart = 1/len * (rayStart -aabbMin);
    rayStop = 1/len * (rayStop -aabbMin);

    int steps_aux = 25;
    int steps = int(0.5 + distance(rayStop, rayStart)  * float(steps_aux));
    vec3 step = (rayStop-rayStart) / float(steps);
    vec3 pos = rayStart + 0.5 * step;
    int travel = steps;

    // compute the possible oclusion of other clouds to the direct sun light 
    vec3 color = vec3(0.0, 0.0, 0.0);
    vec3 l_sun = vec3(1, 0.0, 0.0) ; //skyColor(step_pos); 
    color = l_sun; 

    float Tr = 1; 
    for (travel = 0; travel != steps; travel++) {
        float density = getDensity(pos);

        if(density > 0 ){
            /*---   Transmittance    ---*/
            float l = length(rayStart - pos); // distance the light will travel through the volume
            float transmittance = Transmittance(density, l, step, dir_to_sun); 
            
            if(Tr < 0.01) break; 
            Tr *= transmittance ;
            //color *= transmittance ; 
        }
        

        /*Marching towards the sun has a great impact on performance since for 
        every step an extra number of steps has to be taken. 
        In our implementation four steps towards the sun are taken at 
        exponentially increasing steps size (pág 29 tese)*/
        pos += step; //pow(4, travel) * step;
    }

    return l_sun * Tr;
    //return vec3(1);
    //return (color); 
}
//------------------------------------------------------------------------


// -------------------- Funções de Fase para scattering ------------------ */
float phase_functionSchlick(float g , vec3 light, vec3 step_dir) {
    float k = 1.55*g - 0.55*pow(g, 3); 

    // 1 - k^2
    float n = 1 - pow(k , 2); 
    
    // 4pi *(1 + k*cos(x))^2
    float cos_teta = dot(step_dir, light); // cos(x)
    float d = 4*PI *pow(1 + k*cos_teta, 2) + 0.0001; 

    return  float(n /d);
}

/*  Henyey-Greenstein function (Mie Phase Function):
        light : direção da luz
        step_dir : "direção da camara" no percurso do Ray Marching
    f(x) = (1 - g^2) / (4PI * (1 + g^2 - 2g*cos(teta))^[3/2])        */
float phase_functionHG(float g , vec3 dir_to_sun, vec3 rayDirection) {
    float numerador = 1 - g*g; // 1 - g^2
    
    float cos_teta = dot(rayDirection, dir_to_sun); // cos(x)
    float denominador = 1 + g*g - 2*g*cos_teta; // 1 + g^2 - 2g*cos(x)
    
    return float(numerador  / (4*PI * pow(denominador, 1.5)));
}

/*  Cornette-Shank aproach
    f(x) = 3*(1 - g^2) *       (1 + cos^2(teta))
           2*(2 + g^2)   (1 + g^2 -2g*cos(teta^[3/2]))  */
float phase_functionCS(float g, vec3 light, vec3 step_dir) {

    // 3*(1 - g^2) / 2*(2 + g^2)
    float n = (3/2) * (1 - pow(g, 2))/(2+pow(g, 2)); 
    
    // (1 + cos^2(teta)) / (1 + g^2 -2g*cos(teta^[3/2]))
    float cos_teta = dot(step_dir, light); // cos(x)
    float d = 1 + pow(g, 2) - 2*g*cos_teta; 
    return n * (1 + pow(cos_teta, 2)) / pow(d, 1.5);
}

float Scattering(vec3 rayDirection, vec3 dir_to_sun){
    /*  The g parameter varies from −1 < g < 1
            - Back Scattering:  −1 ≤ g < 0 
            - Isotropic scatterin: g = 0
            - Forward Scattering: 0 < g ≤ 1
    */
    float g0 = g0_phase_function;
    float g1 = g1_phase_function;

    float phase_G0, phase_G1; 

    rayDirection = normalize(rayDirection);

    // Henyey-Greenstein function:
    phase_G0 = phase_functionHG(g0, dir_to_sun, rayDirection);
    //phase_G1 = phase_functionHG(g1, dir_to_sun, rayDirection);

    // Cornette-Shank aproach
    //phase_G0 = phase_functionCS(g0, dir_to_sun, rayDirection);
    //phase_G1 = phase_functionCS(g0, dir_to_sun, rayDirection);
    
    // Schlick approximation of HG function 
    //phase_G0 = phase_functionSchlick(g0, dir_to_sun, rayDirection);
    //phase_G1 = phase_functionSchlick(g1, dir_to_sun, rayDirection);

    return phase_G0;

    float phase = mix(phase_G1, phase_G0, phase_mix);
    
    return phase; 
}
//------------------------------------------------------------------------

//------------------------------------------------------------------------
/*   Transmittance Tr is the amount of photos that travels unobstructed between
 two points along a straight line. The transmittance can be calculated using 
 Beer-Lambert’s law     
    l - distance travel by the light in the box
    step_dir - direction of the ray march or sun march step 
             -> used only for juraj formula
 */
float Transmittance(float density, float step_length, vec3 rayDirection, vec3 dir_to_sun){
    float sigmaAbs   = sigmaAbsorption;
    float sigmaScatt = sigmaScattering;
    float sigmaExt   = (sigmaAbs + sigmaScatt); // sigma Extintion 

    
    // Beer's Lambert Law (used by Rurik) 
    float beers = exp( -sigmaExt * step_length * 20);

	// Powder Law 
    // (introduced by schneider to take into acount the collectiong of light)
	float powder = 1.0 - exp( - sigmaExt * step_length *2000);

    // Beers-Powder Fisrt aprroach
    return beers;
    return beers * powder; 
    //return powder;
    
    // proposta de Juraj Palenik master thesis 
    rayDirection = normalize(rayDirection);
    float cos_teta = max(0.0, dot(dir_to_sun, rayDirection)); // cos(x)
    
    float B_exp = exp( -sigmaExt * step_length * 20);
    float P_exp = 1 - ((cos_teta + 1)/2) * exp(- sigmaExt * step_length *3500); 
    return B_exp * P_exp;  
    
}   

void ComputLight(vec3 RayOrigin, vec3 rayDirection, float rayLength, 
                 vec3 pos, vec3 step_vec, float density, 
                 vec3 dir_to_sun, vec3 ambiente_light,
                 out vec3 S, out float Tr){ 

    //---   Transmittance  ---
    // distance the light will travel through the volume
    // -> the step length is between [0..1] therefore we mulply by the real ray length
    float l = length(step_vec) * rayLength; 
    float transmittance = Transmittance(density, l, rayDirection, dir_to_sun); 
    Tr = transmittance; 

    //---   Scattering    ---
    float phase = Scattering(rayDirection, dir_to_sun);

    // pos.y used as a gradient along the heith  
    vec3 ambiente = density * ambiente_light; 

    S = vec3(phase) + ambiente;
    /*       
    //---   Evaluate direct loght ---
    vec3 direct_light =  ambiente_light;// vec3(1.0,1.0, 1.0);
    //computDirectLight(atual_pos, step_vec);
    
    //---   Combine everything   ---
    float sigmaExt = ( sigmaAbsorption + sigmaScattering ) ;  // sigma Extintion 
    S =  (vec3(phase));// + ambiente_light) * sigmaExt;
    */
}

void main() {

    float FocalLength = 1.0/ tan(radians(FOV*0.5));
    vec3 rayDirection;
    rayDirection.xy = 2.0 * gl_FragCoord.xy / WindowSize.xy - 1.0;
    rayDirection.xy *= vec2(RATIO,1);
    rayDirection.z = -FocalLength;
    rayDirection = (vec4(rayDirection, 0) * VM).xyz;

    Ray eye = Ray( RayOrigin, normalize(rayDirection));

    float tnear, tfar;
    bool r = IntersectBox(eye, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;

    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;
    float rayLength = distance(rayStart, rayStop);

    vec3 len = aabbMax - aabbMin;
    rayStart = 1/len * (rayStart -aabbMin);
    rayStop = 1/len * (rayStop -aabbMin);

    int steps = int(0.5 + distance(rayStop, rayStart) * volume_steps);
    vec3 step = (rayStop-rayStart) / float(steps);
    vec3 pos = rayStart + 0.5 * step;
    int travel = steps;
    
    // Sky color needs the position of the pixel in an imaginary quad 
    // witch happens to be the same adjustments made to compute ray direction for one pixel
    vec3 BG_color = vec3(skyColor(rayDirection));

    // Compute the direction througth the sun 
    vec2 sunAnglesRad = vec2(sunAngles.x, sunAngles.y) * vec2(PI/180);
    vec3 dir_to_sun = vec3(cos(sunAnglesRad.y) * sin(sunAnglesRad.x),
                             sin(sunAnglesRad.y),
                            -cos(sunAnglesRad.y) * cos(sunAnglesRad.x));
    dir_to_sun = normalize(dir_to_sun);
    // Compute the ambiente light only once, because its the same for every Ray march step
    vec3 ambiente_light = skyColor(dir_to_sun);

    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    vec3 scatteredLight = vec3(0.0);
    float transmittance = 1;

    // Ray Marching no Volume  
    for (;  travel != 0;  travel--) {
        float density = getDensity(pos);
    
        if(density > 0){
            vec3 Sint;
            vec3 S; 
            float Tr; 

            ComputLight(rayStart, rayDirection, rayLength, pos, step, 
                        density, dir_to_sun, ambiente_light, S, Tr);
            
            transmittance *= Tr; 
            scatteredLight += 0.08*S; 
            color.a = 1;

            if(transmittance < 0.4) break; 
           
            /*
            // analytic conservative light scattering 
            float sigmaExt = (sigmaAbsorption + sigmaScattering)* density;
            float clampedExtinction = max( sigmaExt , 0.0000001) ;


            Sint = (S - S* Tr) / ( clampedExtinction );
            vec3 mistura = mix(Sint, BG_color.rgb, 0.5);
            scatteredLight += transmittance*(Sint + BG_color.rgb);
            scatteredLight += (Sint * transmittance);
            scatteredLight += mistura * transmittance;
            transmittance  *= Tr;

            color.rgb =  scatteredLight;  
            color.a   += transmittance;
             */
        }
        pos += step;
    }
      
    //color.rgb = vec3(1.0)*transmittance + ambiente_light*scatteredLight; //-> Teste phase com fundo branco
    color.rgb = (BG_color + scatteredLight) * transmittance; //-> Teste phase com cor ceu
    
    //color.rgb = vec3(1.0) * transmittance; //-> Teste transmittance fundo branco
    //color.rgb = BG_color * transmittance; //-> Teste transmittance cor ceu 

    // tonemapping operator
    vec3 mapped; 

    // Reinhard: pow( clamp(atmos / (atmos+1.0),0.0,1.0), vec3(1.0/2.2) );
    // gamma ~ 0.7 was found to be a good starting point
    mapped = clamp(color.rgb / (color.rgb + 1.0), 0.0, 1.0);
    vec3 color_reinhard = pow( mapped, vec3(1.0 / gamma) );
    color.rgb = color_reinhard;
    /*
    // Logarithmic: pow( clamp(smoothstep(0.0, 12.0, log2(1.0+atmos)),0.0,1.0), vec3(1.0/2.2) ); 
    // gamma ~ 6.0 was found to be a good parameter
    mapped = clamp(smoothstep(0.0, gamma, log2(1.0 + color.rgb)), 0.0, 1.0);
    vec3 color_log = pow( (mapped), vec3(1.0 / 1.2));

    // tone mapping using whit point 
    float base_point = 250;
    float max_iterations = 512.0;
    vec3 white_point = vec3(base_point * (volume_steps / max_iterations));
    vec3 color_white_point = pow(vec3(1.0) - exp(-color.rgb / white_point), vec3(1.0 / 6.0));    
 
    //color.rgb = color_reinhard;
    color.rgb = color_log;
    color.rgb = mix(color_reinhard, color_white_point, 0.9);
    */
    FragColor = color;
}


/*------------------------------------------------------------------------
------------------------------------------------------------------------*/
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


void initSampling(in float dist, in int div, out float quotient, out float segLength) {

    if (sampling == EXPONENTIAL) {
        quotient =  pow(dist, 1.0/(div));
        //segLength = quotient - 1;
    }
    else { // linear sampling
        segLength = dist/div;
    }
}


void computeSegLength(float quotient, float current, inout float segLength) {

    if (sampling == EXPONENTIAL) {
        segLength = current * quotient - current;
    }
    else { // linear sampling
    }
}

vec3 skyColor(vec3 direction) {
    vec2 sunAnglesRad = vec2(sunAngles.x, sunAngles.y) * vec2(PI/180);
	vec3 sunDir = vec3(cos(sunAnglesRad.y) * sin(sunAnglesRad.x),
							 sin(sunAnglesRad.y),
							-cos(sunAnglesRad.y) * cos(sunAnglesRad.x));
							
	vec3 origin = vec3(0.0, earthRadius+1, 0.0);
	vec3 dir = normalize(direction);
	float dist = distToTopAtmosphere(origin, dir);
			
    float quotient, quotientLight, segLengthLight, segLength;
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

    float current = 1;
    initSampling(dist, divisions, quotient, segLength);
    float height;
    for(int i = 0; i < divisions; ++i) {
        computeSegLength(quotient, current, segLength);
        vec3 samplePos = origin + (current + segLength * 0.5) * dir;
        height = length(samplePos) - earthRadius;
        if (height < 0) {
            return vec3(0.9);
            return vec3(0.1, 0.3, 1.0);
        }
        float hr = exp(-height / Hr) * segLength;
        float hm = exp(-height / Hm) * segLength;
        opticalDepthRayleigh += hr;
        opticalDepthMie += hm;
        
        float distLightRay = distToTopAtmosphere(samplePos, sunDir);
        initSampling(distLightRay, divisionsLightRay, quotientLight, segLengthLight);
        float currentLight = 1;
        float opticalDepthLightR = 0;
        float opticalDepthLightM = 0;
        int j = 0;
        
        for (; j < divisionsLightRay; ++j) {
            computeSegLength(quotientLight, currentLight, segLengthLight);
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
            vec3 tau = fourPI * betaR * (opticalDepthRayleigh + opticalDepthLightR) + 
                       fourPI * 1.1 * betaM *  (opticalDepthMie + opticalDepthLightM);
            vec3 att = exp(-tau);
            rayleigh += att * hr;
            mie += att * hm;
        }

        current += segLength;
    }
    vec3 result = (rayleigh *betaR * phaseR + mie * betaM * phaseM) * 20;
    /*
    if (cosViewSun >= 0.999192306417128873735516482698) {
        result =   exp(-fourPI*opticalDepthRayleigh * betaR - fourPI*opticalDepthMie * betaM)*20 ;
        result *=   exp(- opticalDepthMie * betaM) ;
        
    }*/

    // tone mapping
    vec3 white_point = vec3(1.0);
    result = pow(vec3(1.0) - exp(-result / white_point * exposure), vec3(1.0 / 2.2));

    if(length(result) <= 0 ){
        result = vec3(1.0, 1.0, 1.0);
    }

    return result;
}
