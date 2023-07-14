#include "GLBuffer.h"

GLBuffer::GLBuffer()
{
	glGenVertexArrays(1, &_vao);
	glGenBuffers(1, &_vbo);

	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);
}

GLBuffer::~GLBuffer()
{
	
}

void GLBuffer::SetBufferData(void * data, size_t dataSize)
{
	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);

	glBufferData(GL_ARRAY_BUFFER, dataSize, data, GL_STATIC_DRAW);
}