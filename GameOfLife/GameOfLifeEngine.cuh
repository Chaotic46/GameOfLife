#pragma once

#include <glad/glad.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

class GameOfLifeEngine
{
public:
	GameOfLifeEngine(unsigned int vbo);
	~GameOfLifeEngine();

	void Update();

private:
	cudaGraphicsResource * resource;
};