#version 440

in vec4 position;
out vec3 l_dir;
uniform	mat4 m_view;
uniform	vec4 lDir;	   // global space

uniform mat4 PVM;

void main()
{
    gl_Position = position;
    l_dir = normalize(vec3(m_view * -lDir));
}
