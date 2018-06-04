#version 330

in Data {
	vec3 normal;
	vec3 l_dir;
} DataIn;

out vec4 color;

void main() {
	
	vec3 n = normalize(DataIn.normal);
	
	float intensity = max(0.0, dot(n, DataIn.l_dir));
	color = intensity * vec4(0.420, 0.557, 0.137, 1.0);
}
	