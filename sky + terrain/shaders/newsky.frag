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
uniform vec2 WindowSize;
uniform vec3 RayOrigin;

uniform sampler2D grid;
uniform sampler2D heightMap;

uniform float cloud_starty =  500, cloud_finishy = 756;
uniform float cloud_startx = -512, cloud_finishx = 512;
uniform float cloud_startz = -512, cloud_finishz = 512;

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

out vec4 colorOut;

void main(){
	
    colorOut = vec4(0.2,0.2, 1, 1);
	
	// Calcular a posição do pixel no viewport/janela 
	float angle = tan( radians(FOV*0.5) );
	vec3 camRight = cross(camView.xyz, camUp.xyz);
	// Fazer a deslocação para o espaço do viewport/quad 
	// O nosso objeto é o mundo => position no local space = position global space
	vec2 pos = DataIn.position.xy * vec2(RATIO*angle, angle);
	// Versão Luce: vec3 dir = camView.xyz * camPos.z + camUp.xyz * camPos.y + camRight * camPos.x;
	vec3 dir = camView.xyz + camUp.xyz * pos.y + camRight * pos.x;
	Ray eye = Ray( camPos.xyz, normalize(dir) );
    
    /*float FocalLength = 1.0/ tan(radians(FOV*0.5));
    vec3 rayDirection;
    rayDirection.xy = 2.0 * gl_FragCoord.xy / WindowSize.xy - 1.0;
	rayDirection.xy *= vec2(RATIO,1);
    rayDirection.z = -FocalLength;
    rayDirection = (vec4(rayDirection, 0) * m_modelView).xyz;
    Ray eye = Ray( RayOrigin, normalize(rayDirection) );*/

	vec4 floor = vec4(1,0,0,0); 
	float cos = dot(normalize(vec4(camPos.xyz, 0)), floor);

	//cos2 + sin2 = ; O seno é negativo quando se olha para baixo
	float sin = sqrt(1 - pow(cos, 2));

	//Pitágoras - SOH
	float hipo_cloud_start = cloud_starty/sin;
	float hipo_cloud_finish = cloud_finishy/sin;

	int layers = 128;
	float layer_size = (cloud_finishy - cloud_starty)/layers;
	
	//float samples_total_lenght = hipo_sky - hipo_cloud;
	//float nr_samples = 20; //no futuro depende do angulo: quanto menor for o coseno menor o nr de samples
	//float sample_size = samples_total_lenght/nr_samples;
	

	//ponto inicial no volume = pos_cam + K * vector_direção_normalizada
	// => K = distancia da hipotenusa 
	vec3 ini_p_Vol_Cloud = camPos.xyz + hipo_cloud_start * dir;	
	vec3 ponto = ini_p_Vol_Cloud;

	//Se está a olhar para baixo, termina 
	//dot(dir, vec3(1,0,0)) < 0 ???  
	if((int(ponto.y) < 0) ){
		colorOut = vec4(0,1,0,0);
		return;
	}

	//start point again
	if(int(ponto.y) > cloud_starty){
		for(int i = 0; i < layers; i++){
			if(int(ponto.y) > cloud_finishy){	return;	}
			colorOut += 0.005;
			/* ivec2 texCoord; 
			texCoord.x = int(pos.x * 32 / GridSize);
			texCoord.y =  
			int x = int(pos.x * 32 / GridSize); 
        	int y = int(pos. * 32 / GridSize * 32);
        	int z = int(pos.y * 32 / GridSize);
			
			colorOut += 0.001 * vec4(texture2D(grid, ivec2((y+x/1, z))).rgba) ;
			// ... ToDo magic 

			Se as coordenadas estão em espaço global e sem limites, como vamos
			fazer a conversão de espaço mundo para tamanho da textura ? 
			*/

			// next point - batota 
			ponto = ponto + layer_size * dir;
		}
	}

	
	
}