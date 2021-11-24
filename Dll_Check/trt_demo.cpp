//#include "vld.h"

#include "trt_dll.h"
#include <string>
#include "opencv2/opencv.hpp"

// COCO dataset class names
std::vector<std::string> COCO_names{
	"N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",	"train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
	"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
	"umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
	"N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

int main() {

	std::string str = "hi there?";
	std::vector<char> writable(str.begin(), str.end());
	writable.push_back('\0');
	char* ptr = &writable[0];

	void* handle = trt_create_model(ptr);

	// 4) 입력으로 사용할 이미지 준비하기 (resize & letterbox padding) openCV 사용
	unsigned int maxBatchSize = 1;	// 생성할 TensorRT 엔진파일에서 사용할 배치 사이즈 값 
	static const int INPUT_H = 500;
	static const int INPUT_W = 500;
	static const int INPUT_C = 3;
	int NUM_CLASS = 92;  // include background
	int NUM_QUERIES = 100;
	std::vector<uint8_t> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C, 0);
	cv::Mat ori_img;
	cv::Mat img_r(INPUT_H, INPUT_W, CV_8UC3);
	for (int idx = 0; idx < maxBatchSize; idx++) { // mat -> vector<uint8_t> 
		ori_img = cv::imread("../etc/data/000000039769.jpg");
		cv::resize(ori_img, img_r, img_r.size(), cv::INTER_LINEAR);
		memcpy(input.data(), img_r.data, maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
	}

	//std::vector<float> scores_h(maxBatchSize * NUM_QUERIES * (NUM_CLASS - 1));
	//std::vector<float> boxes_h(maxBatchSize * NUM_QUERIES * 4);
	//std::vector<float*> outputs = { scores_h.data(), boxes_h.data() };
	std::vector<float> outputs(maxBatchSize *(NUM_QUERIES * (NUM_CLASS - 1) + NUM_QUERIES * 4));

	trt_input_data(handle, input.data());
	trt_run_model(handle);
	trt_output_data(handle, outputs.data());

	// 이미지 출력 로직
	//prob [100, 91]
	//box  [100, 4]
	std::vector<std::pair<float, int>> items(NUM_QUERIES);
	int offset = (NUM_CLASS - 1) + 4;
	for (int i = 0; i < NUM_QUERIES; i++) { // 100
		float* pred = outputs.data() + i * offset;
		int label = -1;
		float score = -1;
		for (int j = 0; j < (NUM_CLASS - 1); j++) { // 91
			if (score < pred[j]) {
				label = j + i * offset;
				score = pred[j];
			}
		}
		items[i].first = score;
		items[i].second = label;
	}
	sort(items.rbegin(), items.rend());
	std::vector<std::vector<float>> COLORS = { {0.000, 0.447, 0.741}, {0.850, 0.325, 0.098}, {0.929, 0.694, 0.125},
		{0.494, 0.184, 0.556}, {0.466, 0.674, 0.188}, {0.301, 0.745, 0.933} };

	for (int idx = 0; idx < 5 && items[idx].first > 0.9; idx++) {
		int ind = items[idx].second / offset * 95 + 91;
		int label = items[idx].second % offset;
		float cx = outputs[ind ];
		float cy = outputs[ind + 1];
		float w = outputs[ind  + 2];
		float h = outputs[ind  + 3];
		float x1 = (cx - w / 2.0) * ori_img.cols;
		float y1 = (cy - h / 2.0) * ori_img.rows;
		float x2 = (cx + w / 2.0) * ori_img.cols;
		float y2 = (cy + h / 2.0) * ori_img.rows;

		cv::Rect rec(x1, y1, x2 - x1, y2 - y1);
		cv::Scalar color(int(COLORS[idx%COLORS.size()][2] * 100), int(COLORS[idx%COLORS.size()][1] * 100), int(COLORS[idx%COLORS.size()][0] * 100));
		cv::rectangle(ori_img, rec, color, 1.5);
		cv::putText(ori_img, COCO_names[label].c_str(), cv::Point(rec.x, rec.y - 1), cv::FONT_HERSHEY_PLAIN, 0.8, color, 1.5);
		printf("      %d %4d prob=%.5f %s\n", idx, label, items[idx].first, COCO_names[label].c_str());
	}
	// items [100, 2] (sorted) (label = second%offset, box_location = second/offset) 
	cv::imshow("result", ori_img);
	cv::waitKey(0);

	std::cout << "==================================================" << std::endl;

	trt_clean_resouce(handle);
	return 0;
}