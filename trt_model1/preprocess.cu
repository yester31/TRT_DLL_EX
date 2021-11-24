#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdint.h>
#include <cstdio>
#include <cuda.h>
#include <vector>
#include <iostream>

using namespace std;

// ��ó�� �Լ� 0 (NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1](Normalize))
__global__ void kernel_preprocess_0(
	float* output,				// [N,RGB,H,W]
	const unsigned char* input, // [N,H,W,BGR]
	const int batchSize, const int height, const int width, const int channel,
	const int tcount)
{
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= tcount) return;

	const int w_idx = pos % width;
	int idx = pos / width;
	const int h_idx = idx % height;
	idx /= height;
	const int c_idx = idx % channel;
	const int b_idx = idx / channel;

	int g_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + 2 - c_idx;

	output[pos] = input[g_idx] / 255.f;
}

void preprocess_cu_0(float* output, unsigned char*input, int batchSize, int height, int width, int channel, cudaStream_t stream)
{
	int tcount = batchSize * height * width * channel;
	int block = 512;
	int grid = (tcount - 1) / block + 1;

	kernel_preprocess_0 << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, tcount);
}

// ��ó�� �Լ� 1 (NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1], 
// Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))
__constant__ float constMem_mean_std[6];
__global__ void kernel_preprocess_1(
	float* output,				// [N,RGB,H,W]
	const unsigned char* input, // [N,H,W,BGR]
	const int batchSize, const int height, const int width, const int channel,
	const int tcount)
{
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos >= tcount) return;

	const int w_idx = pos % width;
	int idx = pos / width;
	const int h_idx = idx % height;
	idx /= height;
	const int c_idx = idx % channel;
	const int b_idx = idx / channel;

	int g_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + 2 - c_idx;

	output[pos] = (input[g_idx] / 255.f - constMem_mean_std[c_idx]) / constMem_mean_std[c_idx + 3];
}

void preprocess_cu_1(float* output, unsigned char*input, int batchSize, int height, int width, int channel, std::vector<float> &mean_std, cudaStream_t stream)
{
	int tcount = batchSize * height * width * channel;
	int block = 512;
	int grid = (tcount - 1) / block + 1;

	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start);
	cudaMemcpyToSymbol(constMem_mean_std, mean_std.data(), sizeof(float) * 6);
	kernel_preprocess_1 << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, tcount);
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float time;
	//cudaEventElapsedTime(&time, start, stop);
	//std::cout << "elapsed time :: " << time << std::endl;
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	//elapsed time :: 0.635904 
	//elapsed time :: 0.599040 (cuda constant mem w data transfer)
	//elapsed time :: 0.492544 (cuda constant mem wo data transfer)

}