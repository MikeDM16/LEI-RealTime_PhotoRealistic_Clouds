#version 330

uniform mat4 m_pvm;
uniform mat3 m_normal;
uniform mat4 m_model;

in vec4 position;
in vec3 normal;

out vec3 normalV;
out vec4 posCam;

void main() {

	normalV = normalize(m_normal * normal);
	posCam = m_model * position;
	gl_Position = m_pvm * position;
}