#pragma once
#include "NvInfer.h"
#include <vector>
#include "detr.hpp"
#include "logging.hpp"	
#include <io.h>				// access
#include <fstream>

using namespace nvinfer1;

class trt_dll {
public:
	trt_dll(char* parameters);
	void input_data(const void* inputs);
	void run_model();
	void output_data(void* outputs);
	~trt_dll();

private:
	cudaStream_t stream_;
	IRuntime* runtime_;
	ICudaEngine* engine_;
	IExecutionContext* context_;
	std::vector<void*> buffers_;
	int maxBatchSize_;

	int INPUT_H = 500;
	int INPUT_W = 500;
	int INPUT_C = 3;
	int NUM_CLASS = 92;  // include background
	int NUM_QUERIES = 100;
	int precision_mode = 32; // fp32 : 32, fp16 : 16, int8(ptq) : 8
};

void* __cdecl trt_create_model(char* parameters) { return (void*)(new trt_dll(parameters)); }
void __cdecl trt_input_data(void* dll_handle, const void* inputs) { ((trt_dll*)dll_handle)->input_data(inputs); }
void __cdecl trt_run_model(void* dll_handle) { ((trt_dll*)dll_handle)->run_model(); }
void __cdecl trt_output_data(void* dll_handle, void* outputs) { ((trt_dll*)dll_handle)->output_data(outputs); }
void __cdecl trt_clean_resouce(void* dll_handle) { delete ((trt_dll*)dll_handle); }