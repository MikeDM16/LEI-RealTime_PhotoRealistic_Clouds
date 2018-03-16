#version 330

uniform	mat4 m_view;
uniform mat4 m_modelView;
uniform mat4 m_pvm;
uniform mat4 m_model;
uniform vec4 camPos, camView;
uniform vec4 camUp;
uniform float FOV;
uniform float RATIO;
uniform float PI = 3.1415;

uniform float cloud_starty =  500, cloud_finishy = 756;
uniform float cloud_startx = -512, cloud_finishx = 512;
uniform float cloud_startz = -512, cloud_finishz = 512;

uniform sampler2D grid;

uniform	vec4 diffuse;
uniform	vec4 l_dir;	   // global space

in Data {
	vec4 eye;
	vec2 texCoord;
	vec3 normal;
	vec3 l_dir;
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

void main(){
		vec4 A, B, C, D, E, F, G, H;

	A = vec4(cloud_startx,  cloud_starty, cloud_startz, 1);
	B = vec4(cloud_finishx, cloud_starty, cloud_startz, 1);
	C = vec4(cloud_finishx, cloud_starty, cloud_finishz, 1);
	D = vec4(cloud_startx,  cloud_starty, cloud_finishz, 1);
	E = vec4(cloud_startx,  cloud_finishy, cloud_startz, 1);
	F = vec4(cloud_finishx, cloud_finishy, cloud_startz, 1);
	G = vec4(cloud_finishx, cloud_finishy, cloud_finishz, 1);
	H = vec4(cloud_startx,  cloud_finishy, cloud_finishz, 1);

	// multiplica por 1/2  pq é metade do FOV 
	float angle = tan(FOV * PI / 180 * 0.5);
	vec3 camRight = cross(camView.xyz, camUp.xyz);
	// Fazer a deslocação das coordenadas de textura para o espaço do viewport/quad 
	vec2 pos = DataIn.texCoord * vec2(RATIO*angle, angle);
	//vec3 dir = camView.xyz * camPos.z + camUp.xyz * camPos.y + camRight * camPos.x;
	vec3 dir = camView.xyz + camUp.xyz * pos.y + camRight * pos.x;
	
	Ray eye = Ray( camPos.xyz, normalize(dir) );
    
	//normalizar o eye para depois multiplicar a hipotenusa pelo eye
	vec4 floor = vec4(1,0,0,0); 
	float cos = dot(normalize(vec4(camPos.xyz, 0)), floor);

	//cos2 + sin2 = 1
	//o seno é negativo quando se olha para baixo
	float sin = sqrt(1 - pow(cos, 2));

	//Pitágoras - SOH
	float hipo_cloud_start = cloud_starty/sin;
	float hipo_cloud_finish = cloud_finishy/sin;

	int layers = 128;
	float layer_size = (cloud_finishy - cloud_starty)/layers;
	
	//float samples_total_lenght = hipo_sky - hipo_cloud;
	//float nr_samples = 20; //no futuro depende do angulo: quanto menor for o coseno menor o nr de samples
	//float sample_size = samples_total_lenght/nr_samples;
	
	colorOut = vec4(0);

	//ponto inicial no volume = pos_cam + K * vector_direção_normalizada
	// => K = distancia da hipotenusa 
	vec3 ini_p_Vol_Cloud = camPos.xyz + hipo_cloud_start * dir;	


	//start point again
	vec3 ponto = ini_p_Vol_Cloud; 
	if(int(ponto.y) > cloud_starty){
		for(int i = 0; i < layers; i++){
			if(int(ponto.y) > cloud_finishy){	break;	}
			colorOut += 0.005;
			
			// ... ToDo magic 

			// next point - batota 
			ponto = ponto + layer_size * dir;
		}
	}
	
	
}