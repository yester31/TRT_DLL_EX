#pragma once
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <unordered_map>
#include <map>

using namespace nvinfer1;
enum RESNETTYPE { R18 = 0, R34, R50, R101, R152 };
const std::map<RESNETTYPE, std::vector<int>> num_blocks_per_stage = { {R18, {2, 2, 2, 2}},{R34, {3, 4, 6, 3}},{R50, {3, 4, 6, 3}},{R101, {3, 4, 23, 3}},{R152, {3, 8, 36, 3}} };

class detr
{
public:
	void loadWeights(const std::string file, std::unordered_map<std::string, Weights>& weightMap);
	IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, const std::string& lname, float eps = 1e-5);
	ILayer* BasicStem(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& input, int out_channels, int group_num = 1);
	ITensor* BasicBlock(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& input, int in_channels, int out_channels, int stride = 1);
	ITensor* BottleneckBlock(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& input, int in_channels, int bottleneck_channels, int out_channels, int stride = 1, int dilation = 1, int group_num = 1);
	ITensor* MakeStage(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& input, int stage, RESNETTYPE resnet_type, int in_channels, int bottleneck_channels, int out_channels, int first_stride = 1, int dilation = 1);
	ITensor* BuildResNet(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, RESNETTYPE resnet_type, int stem_out_channels, int bottleneck_channels, int res2_out_channels, int res5_dilation = 1);
	ITensor* PositionEmbeddingSine(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, int num_pos_feats = 64, int temperature = 10000);
	ITensor* MultiHeadAttention(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& query, ITensor& key, ITensor& value, int embed_dim = 256, int num_heads = 8);
	ITensor* LayerNorm(INetworkDefinition *network, ITensor& input, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, int d_model = 256);
	ITensor* TransformerEncoderLayer(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& src, ITensor& pos, int d_model = 256, int nhead = 8, int dim_feedforward = 2048);
	ITensor* TransformerEncoder(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& src, ITensor& pos, int num_layers = 6);
	ITensor* TransformerDecoderLayer(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& tgt, ITensor& memory, ITensor& pos, ITensor& query_pos, int d_model = 256, int nhead = 8, int dim_feedforward = 2048);
	ITensor* TransformerDecoder(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& tgt, ITensor& memory, ITensor& pos, ITensor& query_pos, int num_layers = 6, int d_model = 256, int nhead = 8, int dim_feedforward = 2048);
	ITensor* Transformer(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& src, ITensor& pos_embed, int num_queries = 100, int num_encoder_layers = 6, int num_decoder_layers = 6, int d_model = 256, int nhead = 8, int dim_feedforward = 2048);
	ITensor* MLP(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& src, int num_layers = 3, int hidden_dim = 256, int output_dim = 4);
	std::vector<ITensor*> Predict(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, ITensor* src);
	void createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName);

	int INPUT_H = 500;
	int INPUT_W = 500;
	int precision_mode = 32; // fp32 : 32, fp16 : 16, int8(ptq) : 8
	const int INPUT_C = 3;
	const int NUM_CLASS = 92;  // include background
	const int NUM_QUERIES = 100;

	const float SCALING = 0.17677669529663687;
	const float SCALING_ONE = 1.0;
	const float SHIFT_ZERO = 0.0;
	const float POWER_TWO = 2.0;
	const float EPS = 0.00001;
	const int D_MODEL = 256;
	const int NHEAD = 8;
	const int DIM_FEEDFORWARD = 2048;
	const int NUM_ENCODE_LAYERS = 6;
	const int NUM_DECODE_LAYERS = 6;
	const float SCORE_THRESH = 0.5;
	const char* INPUT_BLOB_NAME = "images";
	const std::vector<std::string> OUTPUT_NAMES = { "scores", "boxes" };
	std::string weight_path_ = "../etc/detr.wts";
};

