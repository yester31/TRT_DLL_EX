#include "detr.hpp"
#include <assert.h>			// assert
#include "preprocess.hpp"	// preprocess plugin 
//#include "calibrator.h"		// ptq

using namespace nvinfer1;

void detr::createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName) {
	INetworkDefinition* network = builder->createNetworkV2(0U);

	std::unordered_map<std::string, Weights> weightMap;
	//loadWeights("../DETR_py/detr.wts", weightMap);
	loadWeights(weight_path_, weightMap);

	// build network
	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ INPUT_C, INPUT_H, INPUT_W });
	assert(data);

	Preprocess preprocess{ maxBatchSize, INPUT_C, INPUT_H, INPUT_W, 1, {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225} };// Custom(preprocess) plugin 사용하기
	IPluginCreator* preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");// Custom(preprocess) plugin을 global registry에 등록 및 plugin Creator 객체 생성
	IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection*)&preprocess);// Custom(preprocess) plugin 생성
	IPluginV2Layer* preprocess_layer = network->addPluginV2(&data, 1, *preprocess_plugin);// network 객체에 custom(preprocess) plugin을 사용하여 custom(preprocess) 레이어 추가
	preprocess_layer->setName("[preprocess_layer]"); // layer 이름 설정

	// backbone
	auto features = BuildResNet(network, weightMap, *preprocess_layer->getOutput(0), R50, 64, 64, 256);
	ITensor* pos_embed = PositionEmbeddingSine(network, weightMap, *features, 128, 10000);

	auto input_proj = network->addConvolutionNd(*features, D_MODEL, DimsHW{ 1, 1 }, weightMap["input_proj.weight"], weightMap["input_proj.bias"]);
	assert(input_proj);
	input_proj->setStrideNd(DimsHW{ 1, 1 });
	auto flatten = network->addShuffle(*input_proj->getOutput(0));
	assert(flatten);
	flatten->setReshapeDimensions(Dims4{ input_proj->getOutput(0)->getDimensions().d[0], -1, 1, 1 });
	flatten->setSecondTranspose(Permutation{ 1, 0, 2, 3 });
	auto out1 = Transformer(network, weightMap, "transformer", *flatten->getOutput(0), *pos_embed, NUM_QUERIES, NUM_ENCODE_LAYERS, NUM_DECODE_LAYERS, D_MODEL, NHEAD, DIM_FEEDFORWARD);

	auto class_embed = network->addFullyConnected(*out1, NUM_CLASS, weightMap["class_embed.weight"], weightMap["class_embed.bias"]);
	assert(class_embed);
	auto class_softmax = network->addSoftMax(*class_embed->getOutput(0));
	assert(class_softmax);
	class_softmax->setAxes(2);
	ITensor* softmax_t = class_softmax->getOutput(0);

	auto shuffle_l = network->addShuffle(*softmax_t); // [100,92, 1, 1] - > [100,92]
	shuffle_l->setReshapeDimensions(Dims2(softmax_t->getDimensions().d[0], softmax_t->getDimensions().d[1]));
	ITensor* shuffle_t = shuffle_l->getOutput(0);

	auto slice = network->addSlice(*shuffle_t, Dims2(0, 0), Dims2(shuffle_t->getDimensions().d[0], shuffle_t->getDimensions().d[1] - 1), Dims2(1, 1));
	ITensor* slice_t = slice->getOutput(0); // [100,92] ==> [100,91]

	ITensor* bbox = MLP(network, weightMap, "bbox_embed.layers", *out1);
	auto bbox_sig = network->addActivation(*bbox, ActivationType::kSIGMOID);
	assert(bbox_sig);
	ITensor* bbox_sig_t = bbox_sig->getOutput(0);

	//std::vector<ITensor*> results = { slice_t,  bbox_sig_t };

	IShuffleLayer* shuffle_l2 = network->addShuffle(*bbox_sig_t); // [100, 4, 1, 1] - > [100, 41]
	shuffle_l2->setReshapeDimensions(Dims2{ bbox_sig_t->getDimensions().d[0], bbox_sig_t->getDimensions().d[1] });
	ITensor* shuffle_t2 = shuffle_l2->getOutput(0);

	std::vector<ITensor*> results = { slice_t, shuffle_t2};

	IConcatenationLayer* concat_l = network->addConcatenation(results.data(), (int)results.size());// [100, 91] + [100, 4] -> [100, 95]
	concat_l->setAxis(1);
	ITensor* result_t = concat_l->getOutput(0);

	network->markOutput(*result_t);
	result_t->setName("outputs");
	// build output
	//for (int i = 0; i < results.size(); i++) {
	//	network->markOutput(*results[i]);
	//	results[i]->setName(OUTPUT_NAMES[i].c_str());
	//}

	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	//config->setMaxWorkspaceSize(18 * (1 << 23));  // 18MB
	config->setMaxWorkspaceSize(28 * (1 << 23));  // 28MB
	//config->setMaxWorkspaceSize(1ULL << 30);
	//config->clearFlag(BuilderFlag::kTF32);

	if (precision_mode == 16) {
		std::cout << "==== precision f16 ====" << std::endl << std::endl;
		config->setFlag(BuilderFlag::kFP16);
	}
	else if (precision_mode == 8) {
		/*std::cout << "==== precision int8 ====" << std::endl << std::endl;
		std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
		assert(builder->platformHasFastInt8());
		config->setFlag(BuilderFlag::kINT8);
		Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(maxBatchSize, INPUT_W, INPUT_H, 1, "../data_calib2/", "detr_int8_calib300.table", INPUT_BLOB_NAME);
		config->setInt8Calibrator(calibrator);*/
	}
	else {
		std::cout << "==== precision f32 ====" << std::endl << std::endl;
	}

	std::cout << "Building engine, please wait for a while..." << std::endl;
	IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);
	std::cout << "==== model build done ====" << std::endl << std::endl;

	std::cout << "==== model selialize start ====" << std::endl << std::endl;
	std::ofstream p(engineFileName, std::ios::binary);
	if (!p) {
		std::cerr << "could not open plan output file" << std::endl << std::endl;
		//return -1;
	}
	p.write(reinterpret_cast<const char*>(engine->data()), engine->size());
	std::cout << "==== model selialize done ====" << std::endl << std::endl;
	p.close();
	engine->destroy();
	network->destroy();

	// Release host memory
	for (auto& mem : weightMap)
	{
		free((void*)(mem.second.values));
	}
}

void detr::loadWeights(const std::string file, std::unordered_map<std::string, Weights>& weightMap) {
	std::cout << "Loading weights: " << file << std::endl << std::endl;

	// Open weights file
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");
	
	// Read number of weight blobs
	int32_t count;
	input >> count;
	assert(count > 0 && "Invalid weight map file.");

	while (count--) {
		Weights wt{ DataType::kFLOAT, nullptr, 0 };
		uint32_t size;

		// Read name and type of blob
		std::string name;
		input >> name >> std::dec >> size;
		wt.type = DataType::kFLOAT;

		// Load blob
		uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
		for (uint32_t x = 0, y = size; x < y; ++x) {
			input >> std::hex >> val[x];
		}
		wt.values = val;

		wt.count = size;
		weightMap[name] = wt;
	}
	input.close();
	
}

IScaleLayer* detr::addBatchNorm2d(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, const std::string& lname, float eps) {
	float *gamma = (float*)(weightMap[lname + ".weight"].values);
	float *beta = (float*)(weightMap[lname + ".bias"].values);
	float *mean = (float*)(weightMap[lname + ".running_mean"].values);
	float *var = (float*)(weightMap[lname + ".running_var"].values);
	int len = weightMap[lname + ".running_var"].count;

	float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval[i] = gamma[i] / sqrt(var[i] + eps);
	}
	Weights scale{ DataType::kFLOAT, scval, len };

	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
	}
	Weights shift{ DataType::kFLOAT, shval, len };

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ DataType::kFLOAT, pval, len };

	weightMap[lname + ".scale"] = scale;
	weightMap[lname + ".shift"] = shift;
	weightMap[lname + ".power"] = power;
	IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
	assert(scale_1);
	//free(scval);
	//free(shval);
	//free(pval);

	return scale_1;
}

ILayer* detr::BasicStem(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& input, int out_channels, int group_num) {
	// conv1
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, out_channels, DimsHW{ 7, 7 }, weightMap[lname + ".conv1.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ 2, 2 });
	conv1->setPaddingNd(DimsHW{ 3, 3 });
	conv1->setNbGroups(group_num);

	auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1");
	assert(bn1);

	auto r1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(r1);

	auto max_pool2d = network->addPoolingNd(*r1->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
	max_pool2d->setStrideNd(DimsHW{ 2, 2 });
	max_pool2d->setPaddingNd(DimsHW{ 1, 1 });
	auto mp_dim = max_pool2d->getOutput(0)->getDimensions();
	return max_pool2d;
}

ITensor* detr::BasicBlock(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& input, int in_channels, int out_channels, int stride) {
	// conv1
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, out_channels, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ stride, stride });
	conv1->setPaddingNd(DimsHW{ 1, 1 });

	auto r1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
	assert(r1);

	// conv2
	IConvolutionLayer* conv2 = network->addConvolutionNd(*r1->getOutput(0), out_channels, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	assert(conv2);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });

	// shortcut
	ITensor* shortcut_value = nullptr;
	if (in_channels != out_channels) {
		auto shortcut = network->addConvolutionNd(input, out_channels, DimsHW{ 1, 1 }, weightMap[lname + ".shortcut.weight"], weightMap[lname + ".shortcut.bias"]);
		assert(shortcut);
		shortcut->setStrideNd(DimsHW{ stride, stride });
		shortcut_value = shortcut->getOutput(0);
	}
	else {
		shortcut_value = &input;
	}

	// add
	auto ew = network->addElementWise(*conv2->getOutput(0), *shortcut_value, ElementWiseOperation::kSUM);
	assert(ew);

	auto r3 = network->addActivation(*ew->getOutput(0), ActivationType::kRELU);
	assert(r3);

	return r3->getOutput(0);
}

ITensor* detr::BottleneckBlock(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& input, int in_channels, int bottleneck_channels, int out_channels, int stride, int dilation, int group_num) {
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	// conv1
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, bottleneck_channels, DimsHW{ 1, 1 }, weightMap[lname + ".conv1.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(group_num);

	auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1");
	assert(bn1);

	auto r1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(r1);

	// conv2
	IConvolutionLayer* conv2 = network->addConvolutionNd(*r1->getOutput(0), bottleneck_channels, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], emptywts);
	assert(conv2);
	conv2->setStrideNd(DimsHW{ stride, stride });
	conv2->setPaddingNd(DimsHW{ 1 * dilation, 1 * dilation });
	conv2->setDilationNd(DimsHW{ dilation, dilation });
	conv2->setNbGroups(group_num);

	auto bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2");
	assert(bn2);

	auto r2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
	assert(r2);

	// conv3
	IConvolutionLayer* conv3 = network->addConvolutionNd(*r2->getOutput(0), out_channels, DimsHW{ 1, 1 }, weightMap[lname + ".conv3.weight"], emptywts);
	assert(conv3);
	conv3->setStrideNd(DimsHW{ 1, 1 });
	conv3->setNbGroups(group_num);

	auto bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".bn3");
	assert(bn3);

	// shortcut
	ITensor* shortcut_value = nullptr;
	if (in_channels != out_channels) {
		auto shortcut = network->addConvolutionNd(input, out_channels, DimsHW{ 1, 1 }, weightMap[lname + ".downsample.0.weight"], emptywts);
		assert(shortcut);
		shortcut->setStrideNd(DimsHW{ stride, stride });
		shortcut->setNbGroups(group_num);

		auto shortcut_bn = addBatchNorm2d(network, weightMap, *shortcut->getOutput(0), lname + ".downsample.1");
		assert(shortcut_bn);
		shortcut_value = shortcut_bn->getOutput(0);
	}
	else {
		shortcut_value = &input;
	}

	// add
	auto ew = network->addElementWise(*bn3->getOutput(0), *shortcut_value, ElementWiseOperation::kSUM);
	assert(ew);

	auto r3 = network->addActivation(*ew->getOutput(0), ActivationType::kRELU);
	assert(r3);

	return r3->getOutput(0);
}

ITensor* detr::MakeStage(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& input, int stage, RESNETTYPE resnet_type, int in_channels, int bottleneck_channels, int out_channels, int first_stride, int dilation) {
	ITensor* out = &input;
	for (int i = 0; i < stage; i++) {
		std::string layerName = lname + "." + std::to_string(i);
		int stride = i == 0 ? first_stride : 1;

		if (resnet_type == R18 || resnet_type == R34)
			out = BasicBlock(network, weightMap, layerName, *out, in_channels, out_channels, stride);
		else
			out = BottleneckBlock(network, weightMap, layerName, *out, in_channels, bottleneck_channels, out_channels, stride, dilation);

		in_channels = out_channels;
	}
	return out;
}

ITensor* detr::BuildResNet(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, RESNETTYPE resnet_type, int stem_out_channels, int bottleneck_channels, int res2_out_channels, int res5_dilation) {
	assert(res5_dilation == 1 || res5_dilation == 2);  // "res5_dilation must be 1 or 2"
	if (resnet_type == R18 || resnet_type == R34) {
		assert(res2_out_channels == 64);  // "res2_out_channels must be 64 for R18/R34")
		assert(res5_dilation == 1);  // "res5_dilation must be 1 for R18/R34")
	}

	int out_channels = res2_out_channels;
	ITensor* out = nullptr;
	// stem
	auto stem = BasicStem(network, weightMap, "backbone.0.body", input, stem_out_channels);
	out = stem->getOutput(0);

	// res
	for (int i = 0; i < 4; i++) {
		int dilation = (i == 3) ? res5_dilation : 1;
		int first_stride = (i == 0 || (i == 3 && dilation == 2)) ? 1 : 2;
		out = MakeStage(network, weightMap, "backbone.0.body.layer" + std::to_string(i + 1), *out, num_blocks_per_stage.at(resnet_type)[i], resnet_type, stem_out_channels, bottleneck_channels, out_channels, first_stride, dilation);
		stem_out_channels = out_channels;
		bottleneck_channels *= 2;
		out_channels *= 2;
	}
	return out;
}


ITensor* detr::PositionEmbeddingSine(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, int num_pos_feats, int temperature) {
	// refer to https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py#12
	// TODO: improve this implementation
	auto mask_dim = input.getDimensions();
	int h = mask_dim.d[1], w = mask_dim.d[2];
	std::vector<std::vector<float>> y_embed(h);
	for (int i = 0; i < h; i++)
		y_embed[i] = std::vector<float>(w, i + 1);
	std::vector<float> sub_embed(w, 0);
	for (int i = 0; i < w; i++)
		sub_embed[i] = i + 1;
	std::vector<std::vector<float>> x_embed(h, sub_embed);

	// normalize
	float eps = 1e-6, scale = 6.2831853071;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			y_embed[i][j] = y_embed[i][j] / (h + eps) * scale;
			x_embed[i][j] = x_embed[i][j] / (w + eps) * scale;
		}
	}

	// dim_t
	std::vector<float> dim_t(num_pos_feats, 0);
	for (int i = 0; i < num_pos_feats; i++) {
		dim_t[i] = pow(temperature, (2 * (i / 2) / static_cast<float>(num_pos_feats)));
	}

	// pos_x, pos_y
	std::vector<std::vector<std::vector<float>>> pos_x(h, std::vector<std::vector<float>>(w, std::vector<float>(num_pos_feats, 0)));
	std::vector<std::vector<std::vector<float>>> pos_y(h, std::vector<std::vector<float>>(w, std::vector<float>(num_pos_feats, 0)));
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int k = 0; k < num_pos_feats; k++) {
				float value_x = x_embed[i][j] / dim_t[k];
				float value_y = y_embed[i][j] / dim_t[k];
				if (k & 1) {
					pos_x[i][j][k] = std::cos(value_x);
					pos_y[i][j][k] = std::cos(value_y);
				}
				else {
					pos_x[i][j][k] = std::sin(value_x);
					pos_y[i][j][k] = std::sin(value_y);
				}
			}
		}
	}

	// pos
	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * h * w * num_pos_feats * 2));
	float *pNext = pval;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int k = 0; k < num_pos_feats; k++) {
				*pNext = pos_y[i][j][k];
				++pNext;
			}
			for (int k = 0; k < num_pos_feats; k++) {
				*pNext = pos_x[i][j][k];
				++pNext;
			}
		}
	}
	Weights pos_embed_weight{ DataType::kFLOAT, pval, h * w * num_pos_feats * 2 };
	weightMap["pos"] = pos_embed_weight;
	auto pos_embed = network->addConstant(Dims4{ h * w, num_pos_feats * 2, 1, 1 }, pos_embed_weight);
	assert(pos_embed);

	//free(pval);
	return pos_embed->getOutput(0);
}

ITensor* detr::MultiHeadAttention(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& query, ITensor& key, ITensor& value, int embed_dim, int num_heads) {
	int tgt_len = query.getDimensions().d[0];
	int head_dim = embed_dim / num_heads;

	// q
	auto linear_q = network->addFullyConnected(query, embed_dim, weightMap[lname + ".in_proj_weight_q"], weightMap[lname + ".in_proj_bias_q"]);
	assert(linear_q);
	// k
	auto linear_k = network->addFullyConnected(key, embed_dim, weightMap[lname + ".in_proj_weight_k"], weightMap[lname + ".in_proj_bias_k"]);
	assert(linear_k);
	// v
	auto linear_v = network->addFullyConnected(value, embed_dim, weightMap[lname + ".in_proj_weight_v"], weightMap[lname + ".in_proj_bias_v"]);
	assert(linear_v);
	auto scaling_t = network->addConstant(Dims4{ 1, 1, 1, 1 }, Weights{ DataType::kFLOAT, &SCALING, 1 });
	assert(scaling_t);
	auto q_scaling = network->addElementWise(*linear_q->getOutput(0), *scaling_t->getOutput(0), ElementWiseOperation::kPROD);
	assert(q_scaling);

	auto q_shuffle = network->addShuffle(*q_scaling->getOutput(0));
	assert(q_shuffle);
	q_shuffle->setName((lname + ".q_shuffle").c_str());
	q_shuffle->setReshapeDimensions(Dims3{ -1, num_heads, head_dim });
	q_shuffle->setSecondTranspose(Permutation{ 1, 0, 2 });

	auto k_shuffle = network->addShuffle(*linear_k->getOutput(0));
	assert(k_shuffle);
	k_shuffle->setName((lname + ".k_shuffle").c_str());
	k_shuffle->setReshapeDimensions(Dims3{ -1, num_heads, head_dim });
	k_shuffle->setSecondTranspose(Permutation{ 1, 0, 2 });

	auto v_shuffle = network->addShuffle(*linear_v->getOutput(0));
	assert(v_shuffle);
	v_shuffle->setName((lname + ".v_shuffle").c_str());
	v_shuffle->setReshapeDimensions(Dims3{ -1, num_heads, head_dim });
	v_shuffle->setSecondTranspose(Permutation{ 1, 0, 2 });

	auto q_product_k = network->addMatrixMultiply(*q_shuffle->getOutput(0), MatrixOperation::kNONE, *k_shuffle->getOutput(0), MatrixOperation::kTRANSPOSE);
	assert(q_product_k);

	// src_key_padding_mask are all false, so do nothing here
	// see https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/functional/activation.h#826-#839

	auto softmax = network->addSoftMax(*q_product_k->getOutput(0));
	assert(softmax);
	softmax->setAxes(4);

	auto attn_product_v = network->addMatrixMultiply(*softmax->getOutput(0), MatrixOperation::kNONE, *v_shuffle->getOutput(0), MatrixOperation::kNONE);
	assert(attn_product_v);

	auto attn_shuffle = network->addShuffle(*attn_product_v->getOutput(0));
	assert(attn_shuffle);
	attn_shuffle->setName((lname + ".attn_shuffle").c_str());
	attn_shuffle->setFirstTranspose(Permutation{ 1, 0, 2 });
	attn_shuffle->setReshapeDimensions(Dims4{ tgt_len, -1, 1, 1 });

	auto linear_attn = network->addFullyConnected(*attn_shuffle->getOutput(0), embed_dim, weightMap[lname + ".out_proj.weight"], weightMap[lname + ".out_proj.bias"]);
	assert(linear_attn);

	return linear_attn->getOutput(0);
}

ITensor* detr::LayerNorm(INetworkDefinition *network, ITensor& input, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, int d_model) {
	// TODO: maybe a better implementation https://github.com/NVIDIA/TensorRT/blob/master/plugin/common/common.cuh#212
	auto mean = network->addReduce(input, ReduceOperation::kAVG, 2, true);
	assert(mean);

	auto sub_mean = network->addElementWise(input, *mean->getOutput(0), ElementWiseOperation::kSUB);
	assert(sub_mean);

	// implement pow2 with scale
	Weights scale{ DataType::kFLOAT, &SCALING_ONE, 1 };
	Weights shift{ DataType::kFLOAT, &SHIFT_ZERO, 1 };
	Weights power{ DataType::kFLOAT, &POWER_TWO, 1 };
	auto pow2 = network->addScaleNd(*sub_mean->getOutput(0), ScaleMode::kUNIFORM, shift, scale, power, 0);
	assert(pow2);

	auto pow_mean = network->addReduce(*pow2->getOutput(0), ReduceOperation::kAVG, 2, true);
	assert(pow_mean);

	auto eps = network->addConstant(Dims4{ 1, 1, 1, 1 }, Weights{ DataType::kFLOAT, &EPS, 1 });
	assert(eps);

	auto add_eps = network->addElementWise(*pow_mean->getOutput(0), *eps->getOutput(0), ElementWiseOperation::kSUM);
	assert(add_eps);

	auto sqrt = network->addUnary(*add_eps->getOutput(0), UnaryOperation::kSQRT);
	assert(sqrt);

	auto div = network->addElementWise(*sub_mean->getOutput(0), *sqrt->getOutput(0), ElementWiseOperation::kDIV);
	assert(div);

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * d_model));
	for (int i = 0; i < d_model; i++) {
		pval[i] = 1.0;
	}
	Weights norm1_power{ DataType::kFLOAT, pval, d_model };
	weightMap[lname + ".power"] = norm1_power;
	auto affine = network->addScaleNd(*div->getOutput(0), ScaleMode::kCHANNEL, weightMap[lname + ".bias"], weightMap[lname + ".weight"], norm1_power, 1);
	assert(affine);

	//free(pval);
	return affine->getOutput(0);
}

ITensor* detr::TransformerEncoderLayer(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& src, ITensor& pos, int d_model, int nhead, int dim_feedforward) {
	auto pos_embed = network->addElementWise(src, pos, ElementWiseOperation::kSUM);
	assert(pos_embed);
	//return pos_embed->getOutput(0);// 수정 필요

	ITensor* src2 = MultiHeadAttention(network, weightMap, lname + ".self_attn", *pos_embed->getOutput(0), *pos_embed->getOutput(0), src, d_model, nhead);
	//return src2;// 수정 필요 (검증 완료)

	auto shortcut1 = network->addElementWise(*src2, src, ElementWiseOperation::kSUM);
	assert(shortcut1);
	//ITensor* src3 = shortcut1->getOutput(0);
	//return src3;// 수정 필요

	ITensor* norm1 = LayerNorm(network, *shortcut1->getOutput(0), weightMap, lname + ".norm1");

	auto linear1 = network->addFullyConnected(*norm1, dim_feedforward, weightMap[lname + ".linear1.weight"], weightMap[lname + ".linear1.bias"]);
	assert(linear1);

	auto relu = network->addActivation(*linear1->getOutput(0), ActivationType::kRELU);
	assert(relu);

	auto linear2 = network->addFullyConnected(*relu->getOutput(0), d_model, weightMap[lname + ".linear2.weight"], weightMap[lname + ".linear2.bias"]);
	assert(linear2);

	auto shortcut2 = network->addElementWise(*norm1, *linear2->getOutput(0), ElementWiseOperation::kSUM);
	assert(shortcut2);

	ITensor* norm2 = LayerNorm(network, *shortcut2->getOutput(0), weightMap, lname + ".norm2");
	return norm2;
}

ITensor* detr::TransformerEncoder(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& src, ITensor& pos, int num_layers) {
	ITensor* out = &src;
	//num_layers = 1; // 수정 필요
	for (int i = 0; i < num_layers; i++) {
		std::string layer_name = lname + ".layers." + std::to_string(i);
		out = TransformerEncoderLayer(network, weightMap, layer_name, *out, pos);
	}
	return out;
}

ITensor* detr::TransformerDecoderLayer(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& tgt, ITensor& memory, ITensor& pos, ITensor& query_pos, int d_model, int nhead, int dim_feedforward) {
	auto pos_embed = network->addElementWise(tgt, query_pos, ElementWiseOperation::kSUM);
	assert(pos_embed);

	ITensor* tgt2 = MultiHeadAttention(network, weightMap, lname + ".self_attn", *pos_embed->getOutput(0), *pos_embed->getOutput(0), tgt);
	//return tgt2; // 정합성 확인 완료

	auto shortcut1 = network->addElementWise(tgt, *tgt2, ElementWiseOperation::kSUM);
	assert(shortcut1);
	ITensor* norm1 = LayerNorm(network, *shortcut1->getOutput(0), weightMap, lname + ".norm1");

	auto query_embed = network->addElementWise(*norm1, query_pos, ElementWiseOperation::kSUM);
	assert(query_embed);

	auto key_embed = network->addElementWise(memory, pos, ElementWiseOperation::kSUM);
	assert(key_embed);

	ITensor* mha2 = MultiHeadAttention(network, weightMap, lname + ".multihead_attn", *query_embed->getOutput(0), *key_embed->getOutput(0), memory);

	auto shortcut2 = network->addElementWise(*norm1, *mha2, ElementWiseOperation::kSUM);
	assert(shortcut2);

	ITensor* norm2 = LayerNorm(network, *shortcut2->getOutput(0), weightMap, lname + ".norm2");

	auto linear1 = network->addFullyConnected(*norm2, dim_feedforward, weightMap[lname + ".linear1.weight"], weightMap[lname + ".linear1.bias"]);
	assert(linear1);

	auto relu = network->addActivation(*linear1->getOutput(0), ActivationType::kRELU);
	assert(relu);

	auto linear2 = network->addFullyConnected(*relu->getOutput(0), d_model, weightMap[lname + ".linear2.weight"], weightMap[lname + ".linear2.bias"]);
	assert(linear2);

	auto shortcut3 = network->addElementWise(*norm2, *linear2->getOutput(0), ElementWiseOperation::kSUM);
	assert(shortcut3);

	ITensor* norm3 = LayerNorm(network, *shortcut3->getOutput(0), weightMap, lname + ".norm3");

	return norm3;
}

ITensor* detr::TransformerDecoder(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& tgt, ITensor& memory, ITensor& pos, ITensor& query_pos, int num_layers, int d_model, int nhead, int dim_feedforward) {
	ITensor* out = &tgt;
	//std::vector<ITensor*> keeps;
	for (int i = 0; i < num_layers; i++) {
		std::string layer_name = lname + ".layers." + std::to_string(i);
		out = TransformerDecoderLayer(network, weightMap, layer_name, *out, memory, pos, query_pos, d_model, nhead, dim_feedforward);
		//IShuffleLayer* shuffle = network->addShuffle(*norm);
		//shuffle->setReshapeDimensions(Dims3{ 1, norm->getDimensions().d[0], norm->getDimensions().d[1] });
		//ITensor* shuffled = shuffle->getOutput(0);
		//keeps.push_back(norm);
	}
	ITensor* norm = LayerNorm(network, *out, weightMap, lname + ".norm", d_model);
	return norm;
	//IConcatenationLayer* concat = network->addConcatenation(keeps.data(), (int)keeps.size());
	//concat->setAxis(0);
	//ITensor* data = concat->getOutput(0);
	//return data;
}

ITensor* detr::Transformer(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& src, ITensor& pos_embed, int num_queries, int num_encoder_layers, int num_decoder_layers, int d_model, int nhead, int dim_feedforward) {
	auto memory = TransformerEncoder(network, weightMap, lname + ".encoder", src, pos_embed, num_encoder_layers);
	//return memory; // 수정 필요

	// construct tgt
	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * num_queries * d_model));
	for (int i = 0; i < num_queries * d_model; i++) {
		pval[i] = 0.0;
	}
	Weights tgt_weight{ DataType::kFLOAT, pval, num_queries * d_model };
	weightMap[lname + ".tgt_weight"] = tgt_weight;
	auto tgt = network->addConstant(Dims4{ num_queries, d_model, 1, 1 }, tgt_weight);
	assert(tgt);
	// construct query_pos
	auto query_pos = network->addConstant(Dims4{ num_queries, d_model, 1, 1 }, weightMap["query_embed.weight"]);
	assert(query_pos);

	auto out = TransformerDecoder(network, weightMap, lname + ".decoder", *tgt->getOutput(0), *memory, pos_embed, *query_pos->getOutput(0), num_decoder_layers, d_model, nhead, dim_feedforward);
	
	//free(pval);
	return out;
}

ITensor* detr::MLP(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, const std::string& lname, ITensor& src, int num_layers, int hidden_dim, int output_dim) {
	ITensor* out = &src;
	for (int i = 0; i < num_layers; i++) {
		std::string layer_name = lname + "." + std::to_string(i);
		if (i != num_layers - 1) {
			auto fc = network->addFullyConnected(*out, hidden_dim, weightMap[layer_name + ".weight"], weightMap[layer_name + ".bias"]);
			assert(fc);
			auto relu = network->addActivation(*fc->getOutput(0), ActivationType::kRELU);
			assert(relu);
			out = relu->getOutput(0);
		}
		else {
			auto fc = network->addFullyConnected(*out, output_dim, weightMap[layer_name + ".weight"], weightMap[layer_name + ".bias"]);
			assert(fc);
			out = fc->getOutput(0);
		}
	}
	return out;
}

std::vector<ITensor*> detr::Predict(INetworkDefinition *network, std::unordered_map<std::string, Weights>& weightMap, ITensor* src) {
	auto class_embed = network->addFullyConnected(*src, NUM_CLASS, weightMap["class_embed.weight"], weightMap["class_embed.bias"]);
	assert(class_embed);
	auto class_softmax = network->addSoftMax(*class_embed->getOutput(0));
	assert(class_softmax);
	class_softmax->setAxes(2);
	ITensor* bbox = MLP(network, weightMap, "bbox_embed.layers", *src);
	auto bbox_sig = network->addActivation(*bbox, ActivationType::kSIGMOID);
	assert(bbox_sig);
	//std::vector<ITensor*> output = { class_softmax->getOutput(0), bbox_sig->getOutput(0) };
	std::vector<ITensor*> output = { class_embed->getOutput(0), bbox_sig->getOutput(0) };
	return output;
}
