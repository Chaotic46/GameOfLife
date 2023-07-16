#pragma once


class GameOfLifeEngine
{
public:
	GameOfLifeEngine();
	~GameOfLifeEngine();

	void Update();

private:
	cudaGraphicsResource * resource;
};