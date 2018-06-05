#version 440

in vec4 position;

out Data {
	vec3 l_dir;
} DataOut;

uniform	mat4 m_view;
uniform	vec4 lDir;	   // global space

uniform mat4 PVM;

void main()
{
    DataOut.l_dir = normalize(vec3(m_view * -lDir));
    gl_Position = position;
}
