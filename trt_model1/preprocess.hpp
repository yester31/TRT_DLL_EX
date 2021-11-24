#pragma once
#include <common.h>
#include <fstream>

struct Preprocess {
	int N;
	int C;
	int H;
	int W;
	int preproc_type;
	float mean[3];
	float std[3];
};

namespace nvinfer1
{
	class PreprocessPluginV2 : public IPluginV2IOExt
	{
	public:
		PreprocessPluginV2(const Preprocess& arg)
		{
			mPreprocess = arg;
		}

		PreprocessPluginV2(const void* data, size_t length)
		{
			const char* d = static_cast<const char*>(data);
			const char* const a = d;
			mPreprocess = read<Preprocess>(d);
			assert(d == a + length);
		}
		PreprocessPluginV2() = delete;

		virtual ~PreprocessPluginV2() {}

	public:
		int getNbOutputs() const noexcept override
		{
			return 1;
		}

		Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override
		{
			return Dims3(mPreprocess.C, mPreprocess.H, mPreprocess.W); // ��� Tensor�� dimenson shape
		}

		int initialize() noexcept override
		{
			return 0;
		}

		void terminate() noexcept override
		{
		}

		// ���� enqueue �Լ����� �߰��� ����� ������ �ʿ� �ϴٸ� �ʿ��� ������ ũ�⸦ �����ϰ� ����
		// enqueue �Լ����� workspace �����͸� �̿��Ͽ� �� ���� ����
		size_t getWorkspaceSize(int maxBatchSize) const noexcept override
		{
			return 0;
		}

		// plugin�� ����� �����ϴ� �Լ�(���� �ʿ�)
		int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
		{
			uint8_t* input = (uint8_t*)inputs[0];
			float* output = (float*)outputs[0];
			std::vector<float> mean_std(6);
			const int H = mPreprocess.H;
			const int W = mPreprocess.W;
			const int C = mPreprocess.C;
			int p_type = mPreprocess.preproc_type;
			//std::cout << "preproc_type : " << p_type  << std::endl;

			switch (p_type) {
			case 0: // vgg11, resnet18, unet
				void preprocess_cu_0(float* output, unsigned char*input, int batchSize, int height, int width, int channel, cudaStream_t stream);
				preprocess_cu_0(output, input, batchSize, H, W, C, stream);
				break;
			case 1: // detr
				void preprocess_cu_1(float* output, unsigned char*input, int batchSize, int height, int width, int channel, std::vector<float> &mean_std, cudaStream_t stream);
				memcpy(mean_std.data(), mPreprocess.mean, 3 * sizeof(float));
				memcpy(mean_std.data() + 3, mPreprocess.std, 3 * sizeof(float));
				preprocess_cu_1(output, input, batchSize, H, W, C, mean_std, stream);
				break;
			default:
				std::cout << "Unknown";
				break;
			}

			// ��� ����
			//cudaDeviceSynchronize();
			//int count = batchSize * H * W * C;
			//std::cout << "count : " << count << std::endl;
			//std::vector<float> gpuBuffer(count);
			//cudaMemcpy(gpuBuffer.data(), output, gpuBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost);
			//std::ofstream ofs("../Validation_py/trt_0", std::ios::binary);
			//if (ofs.is_open())
			//	ofs.write((const char*)gpuBuffer.data(), gpuBuffer.size() * sizeof(float));
			//ofs.close();
			//std::exit(0);

			// �Է� ����
			//cudaDeviceSynchronize();
			//int count = batchSize * H * W * C;
			//std::cout << "count : " << count << std::endl;
			//std::vector<uint8_t> gpuBuffer(count);
			//cudaMemcpy(gpuBuffer.data(), input, gpuBuffer.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
			//std::ofstream ofs("../Validation_py/trt_1", std::ios::binary);
			//if (ofs.is_open())
			//	ofs.write((const char*)gpuBuffer.data(), gpuBuffer.size() * sizeof(uint8_t));
			//ofs.close();
			//std::exit(0);

			return 0;
		}

		size_t getSerializationSize() const noexcept override
		{
			size_t serializationSize = 0;
			serializationSize += sizeof(mPreprocess);
			return serializationSize;
		}

		void serialize(void* buffer) const noexcept override
		{
			char* d = static_cast<char*>(buffer);
			const char* const a = d;
			write(d, mPreprocess);
			assert(d == a + getSerializationSize());
		}

		void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept override
		{
		}

		//! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
		bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept override
		{
			assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
			bool condition = inOut[pos].format == TensorFormat::kLINEAR;
			condition &= inOut[pos].type != DataType::kINT32;
			condition &= inOut[pos].type == inOut[0].type;
			return condition;
		}
		// ����� ������ Ÿ�� ����
		DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override
		{
			assert(inputTypes && nbInputs == 1);
			return DataType::kFLOAT; //
		}

		// plugin �̸� ���� 
		const char* getPluginType() const noexcept override
		{
			return "preprocess";
		}

		// �ش� plugin ���� �ο�
		const char* getPluginVersion() const noexcept override
		{
			return "1";
		}

		void destroy() noexcept override
		{
			delete this;
		}

		IPluginV2Ext* clone() const noexcept override
		{
			PreprocessPluginV2* plugin = new PreprocessPluginV2(*this);
			return plugin;
		}

		void setPluginNamespace(const char* libNamespace) noexcept override
		{
			mNamespace = libNamespace;
		}

		const char* getPluginNamespace() const noexcept override
		{
			return mNamespace.data();
		}

		bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override
		{
			return false;
		}

		bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override
		{
			return false;
		}

	private:
		template <typename T>
		void write(char*& buffer, const T& val) const
		{
			*reinterpret_cast<T*>(buffer) = val;
			buffer += sizeof(T);
		}

		template <typename T>
		T read(const char*& buffer) const
		{
			T val = *reinterpret_cast<const T*>(buffer);
			buffer += sizeof(T);
			return val;
		}

	private:
		Preprocess mPreprocess;
		std::string mNamespace;
	};

	class PreprocessPluginV2Creator : public IPluginCreator
	{
	public:
		const char* getPluginName() const noexcept override
		{
			return "preprocess";
		}

		const char* getPluginVersion() const noexcept override
		{
			return "1";
		}

		const PluginFieldCollection* getFieldNames() noexcept override
		{
			return nullptr;
		}

		IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
		{
			PreprocessPluginV2* plugin = new PreprocessPluginV2(*(Preprocess*)fc);
			mPluginName = name;
			return plugin;
		}

		IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
		{
			auto plugin = new PreprocessPluginV2(serialData, serialLength);
			mPluginName = name;
			return plugin;
		}

		void setPluginNamespace(const char* libNamespace) noexcept override
		{
			mNamespace = libNamespace;
		}

		const char* getPluginNamespace() const noexcept override
		{
			return mNamespace.c_str();
		}

	private:
		std::string mNamespace;
		std::string mPluginName;
	};
	REGISTER_TENSORRT_PLUGIN(PreprocessPluginV2Creator);
};
