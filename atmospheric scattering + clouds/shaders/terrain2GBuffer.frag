#version 330

in vec3 normalV;
in vec4 posCam;

layout (location = 0) out vec4 pos;
layout (location = 1) out vec4 normal;

void main() {

	normal = vec4(normalize(normalV) * 0.5 + 0.5, 1.0);
	pos = posCam;

}