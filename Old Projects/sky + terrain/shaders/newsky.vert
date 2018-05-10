#version 330

uniform	mat4 m_pvm;			// model -> local
uniform	mat4 m_viewModel;	// model -> camara
uniform	mat4 m_view;		// global -> camara
uniform	mat3 m_normal;

uniform	vec4 l_dir;	   // global space

in vec4 position;	// local space
in vec3 normal;		// local space
in vec2 texCoord0;

// the data to be sent to the fragment shader
out Data {
	vec4 eye;
	vec2 texCoord;
	vec3 normal;
	vec3 l_dir;
	vec4 position;
} DataOut;

void main () {
	DataOut.texCoord = texCoord0;
	DataOut.position = position;
	DataOut.normal = normalize(m_normal * normal);
	DataOut.eye = -(m_viewModel * position);
	DataOut.l_dir = normalize(vec3(m_view * -l_dir));
	gl_Position = m_pvm * position;	
}