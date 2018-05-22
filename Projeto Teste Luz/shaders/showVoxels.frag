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
uniform float threshold_erosion;
uniform float erosion_amount;

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
uniform float sigmaAbsorption;
uniform float sigmaScattering;

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
    //return densidadeR;
    
    float densidadeG = texelFetch(shapeNoise, ivec2(textCoord), level).g;
    float densidadeB = texelFetch(shapeNoise, ivec2(textCoord), level).b;
    float densidadeA = texelFetch(shapeNoise, ivec2(textCoord), level).a;
    return densidadeR * 0.625*densidadeG * 0.25*densidadeB * 0.125*densidadeA; ;
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
    //return densidadeR;
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
    return max(smoothstep(0.0, 0.1, h) - smoothstep(0.7, 1.0, h), 0); // cumulonimbus
}

float HeightGradient(vec3 pos, float h_start, float h_cloud) {
    float atenuacao = 1.0; 

    //Nuvens rasteiras e com pouca altura
    if((h_start < 0.1) && (h_cloud < 0.3) ){
        atenuacao = density_gradient_stratus(pos.y);
    }else
        if((h_start < 0.5) && (h_cloud < 0.6)){
            atenuacao = density_gradient_cumulus(pos.y);
            //atenuacao = texelFetch(shapeNoise, ivec2(textCoord), level).a;
        }else
            atenuacao = density_gradient_cumulonimbus(pos.y);
    return atenuacao * pos.y;//*layer_Height; 
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
    float density = weather.r;

    // Aplicação da função Height signal
    density *= HeightSignal(pos, weather.b, weather.g);

    //--- Fase da Shape  ---
    density *= getShape(pos);

    //--- Fase da Erosion ---
 
    if(density < threshold_erosion)
        density -= erosion_amount *  getErosion(pos);
    
    // Only use positive densitys after erosion !
    if(density > 0)
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
vec4 computDirectLight(vec3 step_pos){
    // Create an Ray from the step position to the light source 
    //vec3 dir_to_sun = vec3(lPosition) - step_pos; 
    vec3 ldir_n = normalize(vec3(m_view * lDir));
    Ray ray_to_sun = Ray( step_pos, normalize(ldir_n) );

    // Same code to adjust the iterations to the box size 
    float tnear, tfar;
    bool r = IntersectBox(ray_to_sun, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;

    vec3 rayStart = ray_to_sun.Origin;
    vec3 rayStop = ray_to_sun.Origin + ray_to_sun.Dir * tfar;

    vec3 len = aabbMax - aabbMin;
    rayStart = 1/len * (rayStart -aabbMin);
    rayStop = 1/len * (rayStop -aabbMin);

    //int steps = int(0.5 + distance(rayStop, rayStart)  * float(GridSize) * 2);
    int steps = 25;
    vec3 step = (rayStop-rayStart) / float(steps);
    vec3 pos = rayStart + 0.5 * step;
    int travel = steps;

    // compute the possible oclusion of other clouds to the direct sun light 
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 l_sun = vec4(lDiffuse.rgb, 1); 

    for (travel = 0; travel != steps; travel++) {
        float density = getDensity(pos);

        if(density > 0 ){
            /*---   Transmittance    ---*/
            float l = length(rayStart - pos); // distance the light will travel through the volume
            float transmittance = Transmittance(density, l, step); 
            
            color +=  l_sun * transmittance;
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
	double pi = 3.14159;

    // 1 - g^2
	float n = 1 - pow(g , 2); 
	
    // 1 + g^2 - 2g*cos(x)
    float cos_teta = dot(light,step_dir); // cos(x)
	float d = 1 + pow(g ,2) - 2 * g * cos_teta; 
    
    return  float(n  / (4*pi * pow(d, 1.5)));
}

/*  Cornette-Shank aproach
    This phase function is also well suited for clouds but is more 
 time consuming to calculate
    f(x) = 3*(1 - g^2) *       (1 + cos^2(teta))
           2*(2 + g^2)   (1 + g^2 -2g*cos(teta^[3/2]))  */
float phase_functionCS(float g, vec3 light, vec3 step_dir) {
	double pi = 3.14159;

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
    float phase_G0 =  10*phase_functionHG(g, dir_to_sun, step_dir);
   
    // Cornette-Shank aproach
    float phase_G1 =  phase_functionCS(g, dir_to_sun, step_dir);

    float phase = mix(phase_G0, phase_G1, 0.5);
    
    //vec4 ambiente_light =  skyColor(step_dir, dir_to_sun, step_pos, g);
    vec4 ambiente_light = vec4(0.1,0.1,0.1,0);
    
    //vec4 sun_light = computDirectLight(step_pos);
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
    float sigmaScatt = sigmaScattering * density; // sigma Scattering 
    
    float sigmaExt = (sigmaAbs + sigmaScatt)  * density; // sigma Extintion 
    
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

vec4 ComputLight(vec3 RayOrigin, vec3 rayDirection, vec3 atual_pos, float density, out vec3 S,out float Tr){ 

    /*---   Transmittance    ---*/
    float l = length(RayOrigin - atual_pos); // distance the light will travel through the volume
    //float transmittance = Transmittance(density, l, rayDirection); 
    Tr = Transmittance(density, l, rayDirection); 

    /*---   Scattering    ---*/
    float g = g_phase_function * density; // sigma scattering   
    vec4 scatter_light = Scattering(g, atual_pos, rayDirection);

    /*---   Light Influence  (ciclo até ao sol)  ---*/
    vec4 intensity =  computDirectLight(atual_pos);
    
    /*---   Combine everything   ---*/
    S = vec3(Tr *scatter_light + intensity); 

    return (Tr *scatter_light + intensity); 

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
    
    //vec3 scattering =  vec3(0.0);
    //float transmittance = 1;
    
    vec4 color = vec4(0.0);
    
    // Ray Marching no Volume  
    for (;  /*color.w == 0  && */ travel != 0;  travel--) {
        
        float density = getDensity(pos);
        
        if(density > 0){
            vec3 S; 
            float Tr; 
            // Como fazer conservative light scattering ? 
            //ComputLight(rayStart, rayDirection, pos, density, S, Tr);
            // sigmaExt = sigmaAbs + sigmaScatt
            //scattering +=  S - S* transmittance / (sigmaExtintion * density + 0.00001);
            //transmittance *= Tr; 
            //color.rgb += scattering;
            //color.a *= transmittance;
           
            color += ComputLight(rayStart, rayDirection, pos, density, S, Tr);
            //color += 0.02 * clamp(ComputLight(rayStart, rayDirection, pos, density), 0.0, 1.0);
         
        }
        pos += step;
    }

    // tonemapping operator
    // Reinhard: pow( clamp(atmos / (atmos+1.0),0.0,1.0), vec3(1.0/2.2) );
    //color = pow( clamp(color/(color + 1.0), 0.0, 1.0), vec4(15.0/2.2) );
    
    // Logarithmic: pow( clamp(smoothstep(0.0, 12.0, log2(1.0+atmos)),0.0,1.0), vec3(1.0/2.2) ); */
    //color = pow( clamp(smoothstep(0.0, 12.0, log2(1.0+color)),0.0,1.0), vec4(1.0/2.2) );

    // tone mapping
	vec4 white_point = vec4(100);
	color = pow(vec4(1.0) - exp(-color / white_point), vec4(1.0 / 2.2));
    color = clamp(color, 0.0, 1.0);

    FragColor.rgb = color.rgb;
    FragColor.a =color.a;
}
