#include "GameOfLifeEngine.cuh"



#include "GameDefs.h"


__global__ void DetermineIfAlive(float * ptr)
{
	int numVerticesPerSquare = 6;
	int numSquaresPerRow     = (NUM_SQUARES_PER_AXIS + 2) * 2;
	int numVerticesPerRow    = numSquaresPerRow * numVerticesPerSquare;
	int tIdx                 = (threadIdx.x  + blockIdx.x * blockDim.x) * numVerticesPerSquare + numVerticesPerRow + numVerticesPerSquare;

	unsigned int count = ptr[tIdx - numVerticesPerSquare]                     + ptr[tIdx + numVerticesPerSquare]                     +
                         ptr[tIdx - numVerticesPerRow]                        + ptr[tIdx + numVerticesPerRow]                        +
                         ptr[tIdx - numVerticesPerRow - numVerticesPerSquare] + ptr[tIdx - numVerticesPerRow + numVerticesPerSquare] +
                         ptr[tIdx + numVerticesPerRow - numVerticesPerSquare] + ptr[tIdx + numVerticesPerRow + numVerticesPerSquare];

	float alive = count == 2 || count == 3 ? 1.0f : 0.0f;
	__syncthreads();
	ptr[tIdx] = alive;
	ptr[tIdx + 1] = alive;
	ptr[tIdx + 2] = alive;
	ptr[tIdx + 3] = alive;
	ptr[tIdx + 4] = alive;
	ptr[tIdx + 5] = alive;
}

GameOfLifeEngine::GameOfLifeEngine(unsigned int vbo)
{
	cudaGraphicsGLRegisterBuffer(&resource, vbo, cudaGraphicsRegisterFlagsNone);
}

GameOfLifeEngine::~GameOfLifeEngine()
{
	
}

void GameOfLifeEngine::Update()
{
	float * devPtr;
	size_t size;

	cudaGraphicsMapResources(1, &resource);

	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

	DetermineIfAlive<<<NUM_SQUARES_PER_AXIS, NUM_SQUARES_PER_AXIS>>>(devPtr);

	cudaGraphicsUnmapResources(1, &resource);
}