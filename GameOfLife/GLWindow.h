#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

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
	void SetupWindow(unsigned int glVersionMajor, unsigned int glVersionMinor);
	void LoadGLAD();

	GLFWwindow * _window;

	unsigned int _width;
	unsigned int _height;
};

