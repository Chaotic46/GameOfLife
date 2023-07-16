#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "GLBuffer.h"
#include "GLShader.h"

#include "GameOfLifeEngine.cuh"

/* \brief Class to create and OpenGL window.
 */
class GLWindow
{
public:
	GLWindow(unsigned int width, unsigned int height, unsigned int glVersionMajor = 3, unsigned int glVersionMinor = 3);
	~GLWindow();

	void Render();

	bool ShouldWindowClose();

private:
	void InitializeLife();
	void SetupWindow(unsigned int glVersionMajor, unsigned int glVersionMinor);
	void LoadGLAD();

	struct GLTriangleVertex
	{
		float v1;
		float v2;
		float v3;
	};

	struct GLTriangle
	{
		GLTriangleVertex p1;
		GLTriangleVertex p2;
		GLTriangleVertex p3;
	};

	struct TriangleColor
	{
		float p1Color;
		float p2Color;
		float p3Color;
	};

	GLFWwindow * _window;

	GLBuffer * _buffer;
	GLShader * _shader;

	GameOfLifeEngine * _engine;

	unsigned int _width;
	unsigned int _height;

	const unsigned int _numElements;
};

