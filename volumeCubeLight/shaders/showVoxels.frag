#version 440

out vec4 FragColor;

uniform vec4 lDiffuse;

in Data {
	vec3 l_dir;
} DataIn;


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
uniform float scatter_coef = 0.5;
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


//------------------ Função Absorção ---------------------------------------
/*  Absorption coefficiente is the probability that a photon is absorbed when 
 traveling through the cloud
    P_abs_coef is an interface parameter for the probability 
    f(x) = e^(-absor_coef_x) * L(x,w)       */
float calcAbsorption(){
    float absor_coef = P_abs_coef; // interface parameter

    float intensidade = 1; // Como 

    return exp(-absor_coef) * intensidade;
}

// -------------------- Funções de Fase para scattering ------------------ */
/*  Henyey-Greenstein function:
        light : direção da luz
        step_dir : "direção da camara" no percurso do Ray Marching
    f(x) = (1 - g^2) / (4PI * (1 + g^2 - 2g*cos(teta))^[3/2])        */
float phase_functionHG(vec3 light, vec3 step_dir) {
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
float phase_functionCS(vec3 light, vec3 step_dir) {
	double pi = 3.14159;

	// 3*(1 - g^2) / 2*(2 + g^2)
	float n = (3/2) * (1 - pow(scatter_coef, 2))/(2+pow(scatter_coef, 2)); 
	
    // (1 + cos^2(teta)) / (1 + g^2 -2g*cos(teta^[3/2]))
    float cos_teta = dot(light,step_dir); // cos(x)
	float d = 1 + pow(scatter_coef,2) - 2*scatter_coef*cos_teta; 
    return n * (1 + pow(cos_teta, 2)) / pow(d, 1.5);
}
float calcScattering(vec3 step_dir){
    vec3 light = vec3(0.4); // uniform com light dir 
    
    // Henyey-Greenstein function:
    return phase_functionHG(light, step_dir);
    // or Cornette-Shank aproach
    return phase_functionCS(light, step_dir);
}
//------------------------------------------------------------------------

// ------------ Funções Extintion e Transmittance ------------------------ 
float calcExctintion(vec3 step_dir){
    float absor_coef = calcAbsorption();
    float scatter_coef = calcScattering(step_dir); 

    // sigma_T = sigma_A + sigma_S
    float trans_coef = absor_coef + scatter_coef; 

    return trans_coef; 
}

/*   Transmittance Tr is the amount of photos that travels unobstructed between
 two points along a straight line. The transmittance can be calculated using
 Beer-Lambert’s law      */
float calcTransmittance(){
    // Calcular integral Tr entre x0 e x1, sumando os trans_coefs de cada step
    float int_trans_coef = 0;

    /*for(x = x0, x != x1, x += step){
        int_trans_coef += calcExctintion(x); 
    }*/

    return exp(-int_trans_coef);
}
//------------------------------------------------------------------------


double calcLight(){
    double sigma = calcAbsorption();
    double transmission = calcTransmittance();

    // integrar

    return 1.0;
}

vec4 simplesLambert(){
    vec3 ld_n = normalize(vec3(6,1,2));
    vec3 n = vec3(0,1,0);

    double intensidade = max( dot(ld_n, n), 0.0);
    vec4 sun_color = vec4(1, 1, 0.9, 1);
    sun_color *= vec4(intensidade);

    return sun_color; 
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
        //vec4 color = vec4(0.2 , 0.5, 1.0, 0.0);
        vec4 color = vec4(0.2 , 0.2, 0.2, 0.0);
    //vec4 color = vec4(0.0);

    for (;  /*color.w == 0  && */ travel != 0;  travel--) {
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

            vec4 l = simplesLambert();
            color += 0.05*l;
            //color += 0.1*vec4(density);
        }

        pos += step;
    }

    FragColor.rgb = vec3(color);
    FragColor.a = color.w;
}
