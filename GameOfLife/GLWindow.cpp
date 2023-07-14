#include "GLWindow.h"

#include <iostream>

const char* vertexShaderSource =
"#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";

const char* fragmentShaderSource =
"#version 330 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
"}\0";

GLWindow::GLWindow(unsigned int width, unsigned int height, unsigned int glVersionMajor, unsigned int glVersionMinor) : _window(NULL),
                                                                                                                        _width(width),
                                                                                                                        _height(height)
{
    float triangle[] =
    {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f,
    };

    SetupWindow(glVersionMajor, glVersionMinor);
    LoadGLAD();

    glViewport(0, 0, _width, _height);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    _shader = new GLShader(vertexShaderSource, fragmentShaderSource);
    _buffer = new GLBuffer();

    _buffer->SetBufferData(triangle, sizeof(triangle));
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

    _buffer->BindBuffer();

    _shader->UseProgram();

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(_window);
    glfwPollEvents();
}