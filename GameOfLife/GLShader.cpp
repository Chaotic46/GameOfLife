#include "GLShader.h"

#include <iostream>

GLShader::GLShader(const char* vertexShader, const char* fragmentShader)
{
	SetShaders(vertexShader, fragmentShader);
}

GLShader::~GLShader()
{

}

void GLShader::SetupShader(GLuint& shader, unsigned int shaderType, const char* source)
{
	int  success = 0;
	char infoLog[512];

	shader = glCreateShader(shaderType);

	glShaderSource(shader, 1, &source, NULL);

	glCompileShader(shader);
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(shader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::" << (shaderType == GL_VERTEX_SHADER ? "VERTEX" : "FRAGMENT") << "::COMPILATION_FAILED\n" << infoLog << std::endl;
		exit(1);
	}
}

void GLShader::SetShaders(const char* vertexShader, const char* fragmentShader)
{
	int  success = 0;
	char infoLog[512];

	SetupShader(_vertex, GL_VERTEX_SHADER, vertexShader);
	SetupShader(_fragment, GL_FRAGMENT_SHADER, fragmentShader);

	_program = glCreateProgram();
	glAttachShader(_program, _vertex);
	glAttachShader(_program, _fragment);
	glLinkProgram(_program);

	glGetProgramiv(_program, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(_program, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINK_FAILED\n" << infoLog << std::endl;
		exit(1);
	}
}