#include "GLWindow.h"

#include <iostream>

GLWindow::GLWindow(unsigned int width, unsigned int height, unsigned int glVersionMajor, unsigned int glVersionMinor) : _window(NULL),
                                                                                                                        _width(width),
                                                                                                                        _height(height)
{
    SetupWindow(glVersionMajor, glVersionMinor);
    LoadGLAD();

    glViewport(0, 0, _width, _height);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

GLWindow::~GLWindow()
{
    glfwTerminate();
}

/* \brief Returns if the user has clicked the exit button on the window.
 */
bool GLWindow::ShouldWindowClose()
{
    return glfwWindowShouldClose(_window);
}

/* \brief Initializes a window using the OpenGL version.
 */
void GLWindow::SetupWindow(unsigned int glVersionMajor, unsigned int glVersionMinor)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, glVersionMajor);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, glVersionMinor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    if (!(_window = glfwCreateWindow(_width, _height, "LearnOpenGL", NULL, NULL)))
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(1);
    }

    glfwMakeContextCurrent(_window);
}

/* \brief Loads GLAD
 */
void GLWindow::LoadGLAD()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        exit(1);
    }
}

/* \brief Performs any rendering to be done.
 */
void GLWindow::Render()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glfwSwapBuffers(_window);
    glfwPollEvents();
}