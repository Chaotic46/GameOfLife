#pragma once

#include <glad/glad.h>

#include <vector>

class GLBuffer
{
public:
	GLBuffer();

	~GLBuffer();

	void BindBuffer() { glBindVertexArray(_vao); }

	void AddAttributeBuffer(unsigned int index, unsigned int numComponents);

	void SetBufferData(unsigned int index, void * data, size_t dataSize);

private:
	GLuint _vao;
	std::vector<GLuint> _vbo;
};