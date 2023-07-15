#include "GLWindow.h"

#include <iostream>

const char* vertexShaderSource =
"#version 330 core\n"
"layout (location = 0) in vec3 shapeCoords;\n"
"layout (location = 1) in float color;\n"
"out vec3 passthrough;\n"
"void main()\n"
"{\n"
"   passthrough = vec3(color, color, color);\n"
"   gl_Position = vec4(shapeCoords.x, shapeCoords.y, shapeCoords.z, 1.0f);\n"
"}\0";

const char* fragmentShaderSource =
"#version 330 core\n"
"in vec3 passthrough;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"    FragColor = vec4(passthrough, 1.0f);\n"
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

    float color[] = 
    {
        1.0f, 1.0f, 1.0f
    };

    SetupWindow(glVersionMajor, glVersionMinor);
    LoadGLAD();

    glViewport(0, 0, _width, _height);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    _shader = new GLShader(vertexShaderSource, fragmentShaderSource);
    
    _buffer = new GLBuffer();
    _buffer->AddAttributeBuffer(0, 3);
    _buffer->AddAttributeBuffer(1, 1);

    _buffer->SetBufferData(0, triangle, sizeof(triangle));
    _buffer->SetBufferData(1, &color,   sizeof(color));
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

    _shader->UseProgram();

    _buffer->BindBuffer();
    
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(_window);
    glfwPollEvents();
}