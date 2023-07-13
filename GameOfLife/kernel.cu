#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GLWindow.h"

#include <stdio.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

#define BLOCK_DIM 10

#define NUM_BLOCKS_X SCREEN_WIDTH / BLOCK_DIM
#define NUM_BLOCKS_Y SCREEN_HEIGHT / BLOCK_DIM

__global__ void GameOfLifeSimple()
{
    unsigned int tIdx   = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
}

int main()
{
    GLWindow shader(800, 600);

    while (!shader.ShouldWindowClose())
    {
        shader.Render();
    }

    return 0;
}