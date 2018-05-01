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
uniform float sigmaScattering;
uniform float sigmaExtintion;
uniform float sigmaAbsorption;

uniform float P_abs_coef = 0.2;


struct Ray {
    vec3 Origin;
    vec3 Dir;
};

struct AABB {
    vec3 Min;
    vec3 Max;
};

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
double HeightSignal(vec3 pos, double h_start, double h_cloud) {
    // h_start = altitude onde inicia a nuvem
    // h_cloud = altura da nuvem, conforme o volume da mesma
    // h_start e h_cloud estão entre [0..1]
    // pos --> retirar a altura onde está o pos na fase a Ray Marching

    // isto se calhar têm todos que ser convertidos para a escala do tamanho da layer

    // Altura da caixa é 3 ...

    double atm = pos.y * layer_Height;
    double r  = (atm - h_start)*(atm - h_start - h_cloud);
    r *= (-4 / (h_cloud * h_cloud + 0.00001));
    return r;

}
//------------------------------------------------------------------------

// -------- Funções de Shape -> gives general form to the cloud  ---------- */
double getShape(vec3 pos){
    // Normalize the coords for the 3D noise size
    vec3 aux = vec3(pos);
    aux.xyz *= shapeHeight;
    aux.x = floor(aux.y)*(shapeHeight/4) + aux.x;
    vec2 textCoord = aux.xz;

    /*
    double r = 0;
    int kernel_size = 2;
    for(int i = -1; i<kernel_size; i++){
        for(int j = -1; i<kernel_size; i++)
            for(int k = -1; i<kernel_size; i++)
                r += texelFetch(shapeNoise, ivec2(textCoord) + ivec2(i +k*shapeHeight, j), level).r;
    }
    //return r / (3*3*3);*/
    
    return texelFetch(shapeNoise, ivec2(textCoord), level).r;
}
//------------------------------------------------------------------------

// -------- Funções de Erosion -> gives details to cloud  ---------------- */
double getErosion(vec3 pos){
    // Normalize the coords for the 3D noise size
    vec3 aux = vec3(pos);
    aux.xyz *= erosionHeight;
    aux.x = floor(aux.y)*erosionHeight + aux.x;
    vec2 textCoord = aux.xz;
    return texture(erosionNoise, vec2(textCoord.x/erosionWidth, pos.z), level).r;
    
    /*
    double r = 0;
    int kernel_size = 2;
    for(int i = -1; i<kernel_size; i++){
        for(int j = -1; i<kernel_size; i++)
            for(int k = -1; i<kernel_size; i++)
                r += texture(erosionNoise, vec2(textCoord.x/erosionWidth, pos.z) + vec2(i/erosionHeight + ((k * erosionHeight) / erosionWidth), j/erosionHeight), level).r;
    }
    return r;
    */
    
    //return texelFetch(erosionNoise, ivec2(textCoord), level).r;
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
    return max(smoothstep(0.0, 0.1, h) - smoothstep(0.7, 1.0, h), 0); // cumulonimbus
}

double HeightGradient(vec3 pos, double h_start, double h_cloud) {
    double atm = pos.y*layer_Height;
    return atm;
}
//------------------------------------------------------------------------

/* ----- Função para determinar a densidade numa determinada posição ----- */
double getDensity(vec3 pos){
    vec3 aux = vec3(pos);
    aux.x *= weatherHeight;
    aux.z *= weatherWidth;
    //aux.x = floor(aux.y)*(weatherHeight) + aux.x;
    vec2 textCoord = aux.xz;

    // Densidade inicial obtida da weather texture
    vec3 weather = texelFetch(weatherTexture, ivec2(textCoord), level).rgb;
    double density = weather.r;

    // Aplicação da função Height signal
    density *= HeightSignal(pos, weather.b, weather.g);

    //--- Fase da Shape  ---
    density *= getShape(pos);

    //--- Fase da Erosion ---
    density -= getErosion(pos);

    // Only use positive densitys after erosion !
    if(density > 0){
        //density *= HeightGradient(pos, weather.b, weather.g);

        // Nuvens rasteiras e com pouca altura
        if((weather.b < 0.1) && (weather.g < 0.3) ){
            density *= density_gradient_stratus(pos.y);
        }else
            if((weather.b < 0.5) && (weather.g < 0.6)){
                density *= density_gradient_cumulus(pos.y);
                //density = texelFetch(shapeNoise, ivec2(textCoord), level).a;
            }else
                density *= density_gradient_cumulonimbus(pos.y);

        // clamp density to 1 for more balance lightning
        if(density > 1)
            density = 1;
    }

    return density;
}
//------------------------------------------------------------------------

//------------------------------------------------------------------------
/* Determinate the direct light coming from the light source and possibly ocluded 
from other clouds 
   - Does another Ray Marching through the volume, but from the step_pos to the sun position
   - Evalute the step_position density in every step of Ray Marching
   - Ultra-Mega fps killer */
vec4 computDirectLight(vec3 step_pos){
    // Create an Ray from the step position to the light source 
    vec3 dir_to_sun = vec3(lPosition) - step_pos; 
    Ray ray_to_sun = Ray( step_pos, normalize(dir_to_sun) );

    // Same code to adjust the iterations to the box size 
    float tnear, tfar;
    bool r = IntersectBox(ray_to_sun, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;

    vec3 rayStart = ray_to_sun.Origin + ray_to_sun.Dir * tnear;
    vec3 rayStop = ray_to_sun.Origin + ray_to_sun.Dir * tfar;

    vec3 len = aabbMax - aabbMin;
    rayStart = 1/len * (rayStart -aabbMin);
    rayStop = 1/len * (rayStop -aabbMin);

    double larg = aabbMax.x - aabbMin.x;
    double cump = aabbMax.z - aabbMin.z;
    double alt  = aabbMax.y - aabbMin.y;

    // !!! More steps sadly kill the performance and leds to a NAU crash  
    // Using 10, 50 or 100 does not change anything visually... but why ? 
    int steps = int(0.5 + distance(rayStop, rayStart)  * float(GridSize) * 2);
    vec3 step = (rayStop-rayStart) / float(steps);
    vec3 pos = rayStart + 0.5 * step;
    int travel = steps;

    // compute the possible oclusion of other clouds to the direct sun light 
    vec4 color = vec4(0,0,0,1);
    //vec4 color = vec4(lDiffuse.rgb, 1);
    vec4 l_sun = vec4(lDiffuse.rgb, 1); 

    for (travel = 0; travel != 6; travel++) {
        double density = getDensity(pos);
        color += 2 * l_sun *  float(density);

        /*Marching towards the sun has a great impact on performance since for 
        every step an extra number of steps has to be taken. 
        In our implementation four steps towards the sun are taken at 
        exponentially increasing steps size (pág 29 tese)*/
        pos += pow(4, travel) * step;
    }

    /* //So, if i knew the normal, can i also use the intensity ? 
    vec3 n = vec3(0,1,0);
    vec3 ld_n = normalize(vec3(m_view * -lDir));
    double intensidade = max( dot(ld_n, n), 0.0);
    vec4 sun_color = vec4(1, 1, 0.9, 1);
    color *= float(intensidade); 
    */ 
    
    return vec4(color.rgba); 
    // starting the color whit vec4(0) and then subtrating the densitys does
    // the same effect (as espected )
    //return vec4(lDiffuse.rgb, 1) - vec4(color.rgb,0); 
}
//------------------------------------------------------------------------


// -------------------- Funções de Fase para scattering ------------------ */
/*  Henyey-Greenstein function (Mie Phase Function):
        light : direção da luz
        step_dir : "direção da camara" no percurso do Ray Marching
    f(x) = (1 - g^2) / (4PI * (1 + g^2 - 2g*cos(teta))^[3/2])        */
float phase_functionHG(float scatter_coef, vec3 light, vec3 step_dir) {
	double pi = 3.14159;

    // 1 - g^2
	float n = 1 - pow(scatter_coef, 2); 
	
    // 1 + g^2 - 2g*cos(x)
    float cos_teta = dot(light,step_dir); // cos(x)
	float d = 1 + pow(scatter_coef,2) - 2*scatter_coef*cos_teta; 
    
    return float(n  / (4*pi * pow(d, 1.5f)));
}
/*  Cornette-Shank aproach
    This phase function is also well suited for clouds but is more 
 time consuming to calculate
    f(x) = 3*(1 - g^2) *       (1 + cos^2(teta))
           2*(2 + g^2)   (1 + g^2 -2g*cos(teta^[3/2]))  */
float phase_functionCS(float scatter_coef, vec3 light, vec3 step_dir) {
	double pi = 3.14159;

	// 3*(1 - g^2) / 2*(2 + g^2)
	float n = (3/2) * (1 - pow(scatter_coef, 2))/(2+pow(scatter_coef, 2)); 
	
    // (1 + cos^2(teta)) / (1 + g^2 -2g*cos(teta^[3/2]))
    float cos_teta = dot(light,step_dir); // cos(x)
	float d = 1 + pow(scatter_coef,2) - 2*scatter_coef*cos_teta; 
    return n * (1 + pow(cos_teta, 2)) / pow(d, 1.5);
}

vec4 calcScattering(float scatter_coef, vec3 step_pos, vec3 dir_to_sun, vec3 step_dir){
    // Henyey-Greenstein function:
    float phase =  phase_functionHG(scatter_coef, dir_to_sun, step_dir);
    // or Cornette-Shank aproach
    //float phase =  phase_functionCS(scatter_coef, dir_to_sun, step_dir);

    vec4 sun_light = computDirectLight(step_pos);
    vec4 ambiente_light = vec4(0.4,0.4,0.4,1);

    // Compute S 
    // Multiply by scatter_coef takes a lot of detail :( 
    // Basicamente toda a cor vem da luz ambiente. why ???? 
    // Deve haver chatice na ComputDirectLight 
    return (sun_light*phase + ambiente_light); //* scatter_coef;

}
//------------------------------------------------------------------------

//------------------ Função Absorção ---------------------------------------
/*  Absorption coefficiente is the probability that a photon is absorbed when 
 traveling through the cloud
    P_abs_coef is an interface parameter for the probability 
    f(x) = e^(-absor_coef_x) * L(x,w)       */
float calcAbsorption(vec3 step_pos, float density){
    float absor_coef = sigmaAbsorption; // interface parameter
    //vec4 sun_light = computDirectLight(step_pos);
    return exp(-absor_coef * density);
}
//------------------------------------------------------------------------

// ------------ Funções Extintion e Transmittance ------------------------ 
// Never used.... 
float calcExctintion(vec3 step_pos, vec3 step_dir, float density){
    // sigma_A
    float absor_coef = calcAbsorption(step_pos, density);
  
    // sigma_S
    float scatter_coef; // ? 

    // sigma_T = sigma_A + sigma_S
    float trans_coef = absor_coef + scatter_coef; 

    return trans_coef; 
}
//------------------------------------------------------------------------

//------------------------------------------------------------------------
/*   Transmittance Tr is the amount of photos that travels unobstructed between
 two points along a straight line. The transmittance can be calculated using
    --> Beer-Lambert’s law     */
float calcTransmittance(float sampleSigmaE, vec3 step_pos, vec3 step_dir){
    // Calcular integral Tr entre x0 e x1, sumando os coeficiente extinção de cada step

    float stepdir = length(step_dir);

    // formula pág 28 tese
    return exp(- sampleSigmaE * stepdir);
}
//------------------------------------------------------------------------


//-- Função para calcular os valores se scattering e transmição numa posição  --
void computeCoefs(double densidade, vec3 step_pos, vec3 step_dir, out vec4 scattering, out float transmittance){
    // sigmaScattering is a interface parameter
    float sampleSigmaS = sigmaScattering * float(densidade);
    vec3 dir_to_sun = normalize(vec3(lPosition) - step_pos); 
    vec3 step_dir_n = normalize(step_dir);

    // Compute S 
    // Multiply by sampleSigmaS takes a lot of detail :( 
    scattering = calcScattering(sampleSigmaS, step_pos, dir_to_sun, step_dir_n);
    
    // Compute Tr 
    float sampleSigmaE = sigmaExtintion * float(densidade);
    transmittance = calcTransmittance(sampleSigmaE, step_pos, step_dir);
}
//------------------------------------------------------------------------

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
    //vec4 color = vec4(0.2 , 0.5, 1.0, 0.0);
    vec4 color = vec4(0.2 , 0.2, 0.2, 0.0);
    //vec4 color = vec4(0.0);

    vec4 scatteredLight = vec4(0,0,0,0); 
    float transmittance = 1; 
    for (;  /*color.w == 0  && */ travel != 0;  travel--) {
        // Evaluate the density in this position, based on the weather, shape 
        // and noise textures
        double density = getDensity(pos);
        
        /*Lighting is evaluated for every sample that returned a density larger 
        than zero, when ray marching through the atmosphere (pág 27 Tese)*/
        if(density > 0){
            vec3 step_dir = vec3(step); 
            
            vec4 scattering; 
            float transmittance_r; 
            computeCoefs(density, pos, step_dir, scattering, transmittance_r);
            
            float sampleSigmaE = sigmaExtintion * float(density);
            float clampE = max(sampleSigmaE, 0.000001);
            // improvement by Energy-conserving analytical integration
            // Pág 38 artigo Frostbite 
            vec4 Sint = (scattering - scattering*transmittance_r) / clampE;
            
            // Evaluate light on the iteration position 
            scatteredLight += transmittance * Sint;
            transmittance *= transmittance_r;
            color += 0.2*scatteredLight;
            
            // jabardar só com iluminação "direta" 
            //vec4 sun_light = computDirectLight(pos);
            //color += 0.02*sun_light;
        }
        pos += step;
    }
    
    FragColor = color;
}
