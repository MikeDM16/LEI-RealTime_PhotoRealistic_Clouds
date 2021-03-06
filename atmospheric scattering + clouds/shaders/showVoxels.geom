#version 440

layout(points) in;
layout(triangle_strip, max_vertices = 24) out;

uniform mat4 PVM;
uniform vec3 aabbMin, aabbMax;

vec4 objCube[8]; // Object space coordinate of cube corner
vec4 ndcCube[8]; // Normalized device coordinate of cube corner
ivec4 faces[6];  // Vertex indices of the cube faces

struct AABB {
    vec3 Min;
    vec3 Max;
};

in Data {
	vec3 l_dir;
    vec2 texCoord;
} Datain[];

out Data {
	vec3 l_dir;
    vec2 texCoord;
} DataOut;

void emit_vert(int vert)
{
    gl_Position = ndcCube[vert];
    EmitVertex();
}

void emit_face(int face)
{
    emit_vert(faces[face][1]); emit_vert(faces[face][0]);
    emit_vert(faces[face][3]); emit_vert(faces[face][2]);
    EndPrimitive();
}

void main()
{
    //DataOut.texCoord = Datain.texCoord;
    //DataOut.l_dir = DataIn.l_dir; 

	//AABB aabb = AABB(vec3(-2.0), vec3(-1));
	
    faces[0] = ivec4(0,1,3,2); faces[1] = ivec4(5,4,6,7);
    faces[2] = ivec4(4,5,0,1); faces[3] = ivec4(3,2,7,6);
    faces[4] = ivec4(0,3,4,7); faces[5] = ivec4(2,1,6,5);

	vec3 halfLength = (aabbMax - aabbMin) * 0.5;
    vec4 P = vec4(aabbMin + halfLength,1);
    vec4 I = vec4(halfLength.x,0,0,0);
    vec4 J = vec4(0,halfLength.y,0,0);
    vec4 K = vec4(0,0,halfLength.z,0);

    objCube[0] = P+K+I+J; objCube[1] = P+K+I-J;
    objCube[2] = P+K-I-J; objCube[3] = P+K-I+J;
    objCube[4] = P-K+I+J; objCube[5] = P-K+I-J;
    objCube[6] = P-K-I-J; objCube[7] = P-K-I+J;

    // Transform the corners of the box:
    for (int vert = 0; vert < 8; vert++)
        ndcCube[vert] = PVM * objCube[vert];

    // Emit the six faces:
    for (int face = 0; face < 6; face++)
        emit_face(face);
}