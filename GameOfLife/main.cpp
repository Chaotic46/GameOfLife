#include "GLWindow.h"
#include "windows.h"
#include "GameDefs.h"

int main()
{
    GLWindow shader(SCREEN_WIDTH, SCREEN_HEIGHT);

    while (!shader.ShouldWindowClose())
    {
        shader.Render();
        Sleep(100);
    }

    return 0;
}