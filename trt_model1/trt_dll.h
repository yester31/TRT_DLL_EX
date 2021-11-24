#pragma once

#define TRTMODEL1_EXPORTS true

#ifdef TRTMODEL1_EXPORTS
#define TRTMODEL1_DECLSPEC __declspec(dllexport)
#else
#define TRTMODEL1_DECLSPEC __declspec(dllimport)
#endif

//extern "C" TRT_MODEL_DECLSPEC void* trt_create_model(char* parameters) { return (void*) new trt_model(parameters); };
//extern "C" TRT_MODEL_DECLSPEC void trt_input_data(void* dll_handle, const void* inputs) { ((trt_model*)dll_handle)->input_data(inputs); };
//extern "C" TRT_MODEL_DECLSPEC void trt_run_model(void* dll_handle) { ((trt_model*)dll_handle)->run_model(); };
//extern "C" TRT_MODEL_DECLSPEC void trt_output_data(void* dll_handle, void* outputs ) { ((trt_model*)dll_handle)->output_data(outputs); };
//extern "C" TRT_MODEL_DECLSPEC void trt_clean_resouce(void* dll_handle) { delete ((trt_model*)dll_handle); };

extern "C" TRTMODEL1_DECLSPEC void* trt_create_model(char* parameters) ;
extern "C" TRTMODEL1_DECLSPEC void trt_input_data(void* dll_handle, const void* inputs);
extern "C" TRTMODEL1_DECLSPEC void trt_run_model(void* dll_handle);
extern "C" TRTMODEL1_DECLSPEC void trt_output_data(void* dll_handle, void* outputs);
extern "C" TRTMODEL1_DECLSPEC void trt_clean_resouce(void* dll_handle);