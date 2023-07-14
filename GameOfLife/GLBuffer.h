#pragma once

#include <glad/glad.h>

class GLBuffer
{
public:
	GLBuffer();
	~GLBuffer();

	void BindBuffer() { glBindVertexArray(_vao); }

	void SetBufferData(void * data, size_t dataSize);

private:
	GLuint _vao;
	GLuint _vbo;
};