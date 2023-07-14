#pragma once

#include <glad/glad.h>

class GLShader
{
public:
	GLShader(const char * vertexShader, const char * fragmentShader);
	~GLShader();

	void UseProgram() { glUseProgram(_program); }

	void SetShaders(const char* vertexShader, const char* fragmentShader);

private:
	void SetupShader(GLuint & shader, unsigned int shaderType, const char * source);

	GLuint _program;
	GLuint _vertex;
	GLuint _fragment;
};

