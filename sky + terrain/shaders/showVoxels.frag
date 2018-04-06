#version 440

out vec4 FragColor;

uniform sampler2D grid;
uniform mat4 VM;
uniform float FOV;
uniform float RATIO;
uniform vec2 WindowSize;
uniform vec3 RayOrigin;
uniform int GridSize = 128;
uniform int level = 0;

struct Ray {
    vec3 Origin;
    vec3 Dir;
};

struct AABB {
    vec3 Min;
    vec3 Max;
};

bool IntersectBox(Ray r, AABB aabb, out float t0, out float t1)
{
    vec3 invR = 1.0 / r.Dir;
    vec3 tbot = invR * (aabb.Min-r.Origin);
    vec3 ttop = invR * (aabb.Max-r.Origin);
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
    vec3 up = vec3(0,250,0);
    AABB aabb = AABB(vec3(-512.0, -64.0, -512.0) + up, vec3(+512.0, +64.0, +512.0) + up);
    
    float tnear, tfar;
    IntersectBox(eye, aabb, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;
    
    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;
    rayStart = 0.5 * (rayStart + 1.0);
    rayStop = 0.5 * (rayStop + 1.0);

	int steps = int(0.5 + distance(rayStop, rayStart)  * 32 * 2);
    vec3 step = (rayStop-rayStart) /float (steps);
    vec3 pos = rayStart + 0.5 * step;
    int travel = steps;

	vec4 color = vec4(0.0, 0.749, 1.0, 0.0);
    //vec4 color = vec4(0);

    for (;  /*color.w == 0  && */ travel != 0;  travel--) {
        vec3 posaux = pos;
        posaux.x *= int(pos.z) % 32;

		color += 0.005*vec4(texelFetch(grid, ivec2(posaux.x, int(posaux.y)%32), level).r);
		pos += step;
     }
	//color = color * 0.01;

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