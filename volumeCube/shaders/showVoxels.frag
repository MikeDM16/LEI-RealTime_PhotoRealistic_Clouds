#version 440

out vec4 FragColor;

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

uniform float layer_Height;

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
    
    // Altura da caixa é 2 ...
    double atm = pos.y*layer_Height;
    double r  = (atm - h_start)*(atm - h_start - h_cloud);
    r *= (-4 / (h_cloud * h_cloud + 0.00001));
    return r; 
}

float density_gradient_stratus(const float h){
    return max(smoothstep(0.00, 0.07, h) - smoothstep(0.07, 0.11, h), 0); // stratus, could be better
}

float density_gradient_cumulus(const float h){
    return max(smoothstep(0.00, 0.22, h) - smoothstep(0.4, 0.62, h), 0); // cumulus
    //return smoothstep(0.3, 0.35, h) - smoothstep(0.425, 0.7, h); // cumulus
}

float density_gradient_cumulonimbus(const float h){
    return smoothstep(0.0, 0.1, h) - smoothstep(0.7, 1.0, h); // cumulonimbus
}

double HeightGradient(vec3 pos, double h_start, double h_cloud) {
    // Altura da caixa é 2 ...
    double atm = pos.y*layer_Height;
    return atm;
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
	vec4 color = vec4(0.2, 0.5, 1.0, 1.0);
    //vec4 color = vec4(0.0);
    
    for (;  /*color.w == 0  && */ travel != 0;  travel--) {
        /* ----------------- Tetativa 1 ----------------- 
        vec3 aux = vec3(pos);
        // Passar todas as coordenas do pos para [0,128]
        aux.xyz *= shapeHeight;
        aux.x = floor(aux.y)*shapeHeight + aux.x;
        vec2 textCoord = aux.xz;
        
        // Assumir que a WeatherTexture é a RGB  e a shape o canal alfa
        vec3 weather = texelFetch(shapeNoise, ivec2(textCoord), level).rgb;
        double density = weather.r;

        // Aplicação da função Height signal
        density *= HeightSignal(pos, weather.b, weather.g);

        //--- Fase da shape ---
        vec3 aux1 = vec3(pos);
        aux1.xyz *= shapeHeight;
        aux1.x = floor(aux1.y)*shapeHeight + aux1.x;
        vec2 textCoord1 = aux1.xz;
        density += texelFetch(shapeNoise, ivec2(textCoord1), level).a;
        
        //--- Fase da Erosion ---
        vec3 aux2 = vec3(pos);  
        aux2.xyz *= erosionHeight;
        aux2.x = floor(aux2.y)*erosionHeight + aux2.x;
        vec2 textCoord2 = aux2.xz;
        density -= texelFetch(erosionNoise, ivec2(textCoord2), level).b;
        
        density *= HeightGradient(pos, weather.b, weather.g);

        if(density > 0){ color += 0.01*vec4(density);  }*/

        

        // ----------------- Tetativa 2 ----------------- 
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

        //--- Fase da shape  ---
        vec3 aux1 = vec3(pos);
        aux1.xyz *= shapeHeight;
        aux1.x = floor(aux1.y)*shapeHeight + aux1.x;
        vec2 textCoord1 = aux1.xz;
        density += texelFetch(shapeNoise, ivec2(textCoord1), level).r;

        //--- Fase da shape + Erosion ---
        vec3 aux2 = vec3(pos);  
        aux2.xyz *= erosionHeight;
        aux2.x = floor(aux2.y)*erosionHeight + aux2.x;
        vec2 textCoord2 = aux2.xz;
        density -=  texelFetch(erosionNoise, ivec2(textCoord2), level).r;
        
        // Only use positive densitys after erosion ! 
        if(density > 0){ 
            //density *= HeightGradient(pos, weather.b, weather.g);
            
            // Nuvens rasteiras e com pouca altura 
            if((weather.b < 0.1) && (weather.g < 0.3) ){ 
                density *= density_gradient_stratus(pos.y); 
            }
            else 
                if((weather.b < 0.5) && (weather.g < 0.6)){ 
                    density *= density_gradient_cumulus(pos.y); 
            }else{
                density *= density_gradient_cumulonimbus(pos.y);
            }
        
            // clamp density to 1 for more balance lightning
            if(density > 1) {density = 1; }

            color += 0.01*vec4(density);  
        }
        
       
                
        //pos = aux;
        pos += step;
    }

    FragColor.rgb = vec3(color);
    FragColor.a = color.w;
}