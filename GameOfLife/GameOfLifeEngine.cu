#include "GameOfLifeEngine.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void DetermineIfAlive(float * ptr)
{
	int tIdx = threadIdx.x + 801;

	unsigned int count = ptr[tIdx - 1]   + ptr[tIdx + 1]   +
                         ptr[tIdx - 800] + ptr[tIdx + 800] +
                         ptr[tIdx - 799] + ptr[tIdx - 801] +
                         ptr[tIdx + 799] + ptr[tIdx + 801];

	ptr[tIdx] = count == 2 || count == 3 ? 1.0f : 0.0f;
}

GameOfLifeEngine::GameOfLifeEngine()
{
	
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

	DetermineIfAlive<<<1, 1>>>(devPtr);

	cudaGraphicsUnmapResources(1, &resource);
}