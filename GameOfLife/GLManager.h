#pragma once

#include "GLBuffer.h"
#include "GLShader.h"

class GLManager
{
public:
	GLManager();
	~GLManager();

private:
	GLBuffer * _buffer;
	GLShader * _shader;
};

