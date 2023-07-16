#include "GLWindow.h"

#include "GameDefs.h"

#include <iostream>
#include <stdlib.h>

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
                                                                                                                        _height(height),
                                                                                                                        _numElements((NUM_SQUARES_PER_AXIS + 2) * (NUM_SQUARES_PER_AXIS + 2) * 2) // Multiply by 2 since we need 2 triangles to make a square
                                                                                                                                                                                                 // Add 2 to each NUM_SQUARES_PER_AXIS since we need to buffer the screen for CUDA
{
    SetupWindow(glVersionMajor, glVersionMinor);
    LoadGLAD();

    glViewport(0, 0, _width, _height);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    _shader = new GLShader(vertexShaderSource, fragmentShaderSource);

    _buffer = new GLBuffer();
    _buffer->AddAttributeBuffer(0, 3);
    _buffer->AddAttributeBuffer(1, 1);

    _buffer->SetBufferData(0, NULL, _numElements * sizeof(GLTriangle));
    _buffer->SetBufferData(1, NULL, _numElements * sizeof(TriangleColor));

    InitializeLife();

    _engine = new GameOfLifeEngine(_buffer->GetBuffer(1));
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

/* \brief Creates a 602 x 802 array of vertices and colors
 * 
 * Although the window size is 800x600, we make the input 802x602.
 * This is so that we can create a border around the play area that
 * CUDA can still read without crashing.
 */
void GLWindow::InitializeLife()
{
    GLTriangle    * triangleGrid = new GLTriangle   [_numElements];
    TriangleColor * color        = new TriangleColor[_numElements];

    memset(triangleGrid, 0, _numElements * sizeof(GLTriangle));
    memset(color,        0, _numElements * sizeof(TriangleColor));

    // Increment value used to determine the position of all the squares
    float inc = 2.0f / NUM_SQUARES_PER_AXIS;

    float y = -1.0f;
    for (unsigned int i = 1; i < NUM_SQUARES_PER_AXIS + 1; i++, y += inc)
    {
        float x = -1.0f;
        for (unsigned int j = 1; j < NUM_SQUARES_PER_AXIS + 1; j++, x += inc)
        {
            int index = 2 * ((NUM_SQUARES_PER_AXIS * i) + j);

            float alive = rand() % 2;

            // Assign each set of 2 triangles the same alive color (used to determine if dead or alive
            color[index] =
            {
                alive, alive, alive
            };

            color[index + 1] =
            {
                alive, alive, alive
            };

            // Create the grid layout
            triangleGrid[index] =
            {
                { x,       y,       0.0f },
                { x,       y + inc, 0.0f },
                { x + inc, y,       0.0f }
            };
            
            triangleGrid[index + 1] =
            {
                { x,       y + inc, 0.0f },
                { x + inc, y + inc, 0.0f },
                { x + inc, y,       0.0f }
            };
        }
    }

    // Push all the initial data to OpenGL
    _buffer->SetBufferData(0, triangleGrid, _numElements * sizeof(GLTriangle));
    _buffer->SetBufferData(1, color,        _numElements * sizeof(TriangleColor));
}

/* \brief Initializes a window using the OpenGL version.
 */
void GLWindow::SetupWindow(unsigned int glVersionMajor, unsigned int glVersionMinor)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, glVersionMajor);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, glVersionMinor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    if (!(_window = glfwCreateWindow(_width, _height, "Game of Life", NULL, NULL)))
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
    
    _engine->Update();

    glDrawArrays(GL_TRIANGLES, 0, _numElements * 3); // Multiple the total number of elements by 3 since we have 3 vertices per triangle

    glfwSwapBuffers(_window);
    glfwPollEvents();
}