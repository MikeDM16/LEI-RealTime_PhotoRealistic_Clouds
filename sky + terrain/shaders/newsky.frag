#version 330

uniform	mat4 m_view;
uniform mat4 m_modelView;
uniform mat4 m_pvm;
uniform mat4 m_model;
uniform vec4 camPosition, camView;
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

out vec4 colorOut;

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

	vec3 camRight = cross(camView.xyz, camUp.xyz);

	float angle = tan(FOV * PI / 180 * 0.5);
	vec2 pos = DataIn.texCoord * vec2(RATIO*angle, angle);
	vec3 dir = camView.xyz * camPosition.z + camUp.xyz * camPosition.y + camRight * camPosition.x;
	

	colorOut = vec4(dir, 1);

}