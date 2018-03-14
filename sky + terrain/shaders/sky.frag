#version 330

uniform	mat4 m_view;
uniform mat4 m_modelView;
uniform mat4 m_pvm;
uniform mat4 m_model;
uniform vec4 camPosition, camDirection;

uniform int sky_height = 500;
uniform int cloud_height = 200;
uniform int sky_length = 512;
uniform int layers = 128;
uniform vec4 sky_color = vec4(0, 0, 1, 1);
uniform float[7] densidades = {0.1, 0.3, 0.5, 0.65, 0.78, 0.8, 0.7};
uniform vec3 floor = vec3(1, 0, 0);

uniform sampler2D grid;

uniform	vec4 diffuse;
uniform	vec4 l_dir;	   // global space

in Data {
	vec4 eye;
	vec2 texCoord;
	vec3 normal;
	vec3 l_dir;
	vec4 position;
} DataIn;


struct Ray {
    vec3 Origin;
    vec3 Dir;
};

struct AABB {
    vec3 Min;
    vec3 Max;
};

out vec4 colorOut;

/*
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
}*/

void main() {
	/*
	// transform light to camera space and normalize it
	vec3 ld = normalize( vec3(m_view * -l_dir) );
	vec3 normal = normalize( DataIn.normal );
	float intensity = max(dot(normal, ld), 0.0);

	vec4 spec = vec4(pow(dot( H, normal), 28));
	*/

	/*
	vec4 RayStart = m_view * camPosition;
	vec4 RayDir = normalize(m_view * camDirection);
	vec3 up = vec3(0,250,0);
	Ray ray = Ray(RayStart.xyz, RayDir.xyz);
	AABB aabb = AABB(vec3(-512.0, -64.0, -512.0) + up, vec3(+512.0, +64.0, +512.0) + up);

    float tnear, tfar;
	IntersectBox(ray, aabb, tnear, tfar);
    
    if (tnear < 0.0) tnear = 0.0;


    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;
    rayStart = 0.5 * (rayStart + 1.0);
    rayStop = 0.5 * (rayStop + 1.0);

	int steps = int(0.5 + distance(rayStop, rayStart)  * float(GridSize) * 2);
    vec3 step = (rayStop-rayStart) /float (steps);
    vec3 pos = rayStart + 0.5 * step;
    int travel = steps;
	*/

	//normalizar o eye para depois multiplicar a hipotenusa pelo eye
	float cos = dot(normalize(DataIn.eye), vec4(floor, 0));
	
	//cos2 + sin2 = 1
	//o seno é negativo quando se olha para baixo
	float sin = sqrt(1 - pow(cos, 2));

	//soh
	float hipo_cloud = cloud_height/sin;
	float hipo_sky = sky_height/sin;

	float layer_size = (sky_height - cloud_height)/layers;
	
	float samples_total_lenght = hipo_sky - hipo_cloud;
	
	float nr_samples = 20; //no futuro depende do angulo: quanto menor for o coseno menor o nr de samples
	float sample_size = samples_total_lenght/nr_samples;

	//ponto + vector = ponto
	//eye também é a posição
	vec3 initial_point_cloud = vec3(DataIn.eye) + hipo_cloud;
	vec3 initial_point_sky = vec3(DataIn.eye) + hipo_sky;

	//float color = hipo_cloud/cloud_height;
	//float color2 = dada pelo samples_total_lenght
	//float color3 = abs(cos);
	//colorOut = vec4(color3, color3, color3, 1);

	vec4 color = vec4(0);

	vec4 p_ini = vec4(initial_point_cloud);
	for (int i = 0; i < 128; i++) {
		int x = int(p_ini.x * 32 / 256); 
        int y = int(p_ini.y * 32 / 128 * 32);
        int z = int(p_ini.z * 32 / 256);

		vec2 texCoord = vec2(x, z);
		int level = int(y);
		texCoord.x += level * 32; // nivel/camada do volume
		color += 0.0005 * vec4(texelFetch(grid, ivec2(texCoord), 0).rgba);
		
		p_ini += ...
		
		if(p_ini.y > initial_point_sky) break; // sai fora volume

	}

	colorOut = vec4(1);

	vec4 pos_mundo = camPosition+camView;
	if (pos_mundo.y > 0)
		colorOut = color;

	//colorOut = m_modelView * DataIn.position;
	
	//float d = distance((DataIn.position).rgb, (m_view * camPosition).rgb);

	//vec4 quad_center = camPosition + camDirection;
	//vec4 pos_mundo = quad_center + DataIn.position;
}