#include "GLWindow.h"

#include "GameDefs.h"

int main()
{
    GLWindow shader(SCREEN_WIDTH, SCREEN_HEIGHT);

    while (!shader.ShouldWindowClose())
    {
        shader.Render();
    }

    return 0;
}