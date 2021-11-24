#include "trt_dll.h"
#include "trt_dll.hpp"
#include "logging.hpp"	
#include <io.h>				// access
#include <fstream>
#include "detr.hpp"

using namespace nvinfer1;

// trt_dll ������
trt_dll::trt_dll(char* parameters) {
	// �� ���� ����

	printf("%s\n", parameters);    // ���ڿ� ���
	sample::Logger gLogger;
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	stream_ = stream;

	detr detr_instance = detr();

	int precision_mode = detr_instance.precision_mode;

	unsigned int maxBatchSize = 1;	// ������ TensorRT �������Ͽ��� ����� ��ġ ������ �� 
	bool serialize = false;			// Serialize ����ȭ ��Ű��(true ���� ���� ����)
	char engineFileName[] = "detr";
	char engine_file_path[256];
	//sprintf(engine_file_path, "../Engine/%s_%d.engine", engineFileName, precision_mode);
	sprintf(engine_file_path, "../etc/%s_%d.engine", engineFileName, precision_mode);

	// 1) engine file ����� 
	// ���� ����� true�� ������ �ٽ� �����
	// ���� ����� false��, engine ���� ������ �ȸ����
	//					   engine ���� ������ ����
	bool exist_engine = false;
	if ((_access(engine_file_path, 0) != -1)) {
		exist_engine = true;
	}

	if (!((serialize == false)/*Serialize ����ȭ ��*/ == (exist_engine == true) /*resnet18.engine ������ �ִ��� ����*/)) {
		std::cout << "===== Create Engine file start =====" << std::endl << std::endl; // ���ο� ���� ����
		IBuilder* builder = createInferBuilder(gLogger);
		IBuilderConfig* config = builder->createBuilderConfig();
		detr_instance.createEngine(maxBatchSize, builder, config, DataType::kFLOAT, engine_file_path); // *** Trt �� ����� ***
		builder->destroy();
		config->destroy();
		std::cout << "===== Create Engine file finish =====" << std::endl << std::endl; // ���ο� ���� ���� �Ϸ�
	}

	// 2) engine file �ε� �ϱ� 
	char *trtModelStream{ nullptr };// ����� ��Ʈ���� ������ ����
	size_t size{ 0 };
	std::cout << "===== Engine file load =====" << std::endl << std::endl;
	std::ifstream file(engine_file_path, std::ios::binary);
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		file.read(trtModelStream, size);
		file.close();
	}
	else {
		std::cout << "[ERROR] Engine file load error" << std::endl;
	}

	// 3) file���� �ε��� stream���� tensorrt model ���� ����
	std::cout << "===== Engine file deserialize =====" << std::endl << std::endl;
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	IExecutionContext* context = engine->createExecutionContext();
	delete[] trtModelStream;

	runtime_ = runtime;
	engine_ = engine;
	context_ = context;

	// prepare data memory
	
	//void *data_d, *scores_d, *boxes_d;
	//CHECK(cudaMalloc(&data_d, maxBatchSize * INPUT_H * INPUT_W * INPUT_C * sizeof(uint8_t)));
	//CHECK(cudaMalloc(&scores_d, maxBatchSize * NUM_QUERIES * (NUM_CLASS - 1) * sizeof(float)));
	//CHECK(cudaMalloc(&boxes_d, maxBatchSize * NUM_QUERIES * 4 * sizeof(float)));
	//std::vector<void*> buffers = { data_d, scores_d, boxes_d };

	void *data_d, *output_d;
	CHECK(cudaMalloc(&data_d, maxBatchSize * INPUT_H * INPUT_W * INPUT_C * sizeof(uint8_t)));
	CHECK(cudaMalloc(&output_d, maxBatchSize * (NUM_QUERIES * (NUM_CLASS - 1) + NUM_QUERIES * 4) * sizeof(float)));

	buffers_.push_back(data_d);
	buffers_.push_back(output_d);

	maxBatchSize_ = maxBatchSize;
}

// trt_dll �Ҹ���
trt_dll::~trt_dll() {
	// �ڿ� ���� ����
	for (int i = 0; i < buffers_.size(); i++) {
		CHECK(cudaFree(buffers_[i]));
	}
	context_->destroy();
	runtime_->destroy();
	engine_->destroy();
	cudaStreamDestroy(stream_);
}

// ������ ���� ����
void trt_dll::input_data(const void* inputs) {

	CHECK(cudaMemcpyAsync(buffers_[0], (char*)inputs, maxBatchSize_ * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_));

}

// �߷� ���� ����
void trt_dll::run_model() {

	context_->enqueue(maxBatchSize_, buffers_.data(), stream_, nullptr);

}

// ��� �ε� ����
void trt_dll::output_data(void* outputs) {

	CHECK(cudaMemcpyAsync((float*)outputs, buffers_[1], maxBatchSize_ * (NUM_QUERIES * (NUM_CLASS - 1) + NUM_QUERIES * 4) * sizeof(float), cudaMemcpyDeviceToHost, stream_));
	//CHECK(cudaMemcpyAsync((float*)outputs + (maxBatchSize_ * NUM_QUERIES * (NUM_CLASS - 1)), buffers_[2], maxBatchSize_ * NUM_QUERIES * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream_));
	cudaStreamSynchronize(stream_);
	//cudaDeviceSynchronize();
}

