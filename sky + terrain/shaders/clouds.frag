#version 330

uniform	mat4 m_view;
uniform mat4 m_model_view;

uniform int sky_height = 500;
uniform int cloud_height = 200;
uniform int layers = 7;
uniform vec4 sky_color = vec4(0, 0, 1, 1);
uniform float[7] densidades = {0.1, 0.3, 0.5, 0.65, 0.78, 0.8, 0.7};
uniform vec3 floor = vec3(1, 0, 0);

uniform	vec4 diffuse;
uniform	vec4 l_dir;	   // global space

in Data {
	vec4 eye;
	vec2 texCoord;
	vec3 normal;
	vec3 l_dir;
} DataIn;

out vec4 colorOut;

void main() {
	/*
	// transform light to camera space and normalize it
	vec3 ld = normalize( vec3(m_view * -l_dir) );
	vec3 normal = normalize( DataIn.normal );
	float intensity = max(dot(normal, ld), 0.0);

	vec4 spec = vec4(pow(dot( H, normal), 28));
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
	float color3 = abs(cos);
	colorOut = vec4(1, 1, 1, 1);
}