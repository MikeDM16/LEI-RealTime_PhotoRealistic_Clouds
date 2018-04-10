#version 440

out vec4 FragColor;

uniform sampler2D grid;
uniform int gridWidth = 16384;
uniform int gridHeight = 128; 
uniform int GridSize;

uniform mat4 VM;
uniform float FOV;
uniform float RATIO;
uniform vec2 WindowSize;
uniform vec3 RayOrigin;
uniform int level = 0;
uniform vec3 aabbMin, aabbMax;

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
    vec3 step = (rayStop-rayStart) /float (steps);
    vec3 pos = rayStart + 0.5 * step;
    int travel = steps;
	vec4 color = vec4(0);
    for (;  /*color.w == 0  && */ travel != 0;  travel--) {
        vec3 aux = pos; 
        // Passar todas as coordenas do pos para [0,128]
        aux.xz *= gridHeight; // *= 128;
        // Converter y para uma das N texturas
        aux.y *= gridWidth; // *= 16384 

        vec2 textCoord = vec2(0);
        textCoord.x = aux.x + aux.y; 
        textCoord.y = aux.z;

		color +=  vec4(texelFetch(grid, ivec2(textCoord), level).rgba) ;
        pos = aux;
        pos += step;
     }
	//color = color* 0.01;

	//if (color != vec4(0))
	//	FragColor.rgb = vec3(0.5);
	//else 
	
	/* float k = length(color);
	if (k > 0.1 && k < 0.2)
		color = k * vec4(0,1,0,1);
	else if (k > 0.2 && k < 0.3)
		color = k * vec4(1,0,1,1);*/
    FragColor.rgb = vec3(color);
    FragColor.a = color.w;
}