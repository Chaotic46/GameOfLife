#include "GLBuffer.h"

GLBuffer::GLBuffer()
{
	glGenVertexArrays(1, &_vao);
}

GLBuffer::~GLBuffer()
{
	
}

void GLBuffer::AddAttributeBuffer(unsigned int index, unsigned int numComponents)
{
	GLuint vbo;
	glGenBuffers(1, &vbo);

	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glVertexAttribPointer(index, numComponents, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(index);

	_vbo.push_back(vbo);
}

void GLBuffer::SetBufferData(unsigned int index, void * data, size_t dataSize)
{
	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo[index]);

	glBufferData(GL_ARRAY_BUFFER, dataSize, data, GL_STATIC_DRAW);
}