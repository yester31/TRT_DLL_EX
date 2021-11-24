import ctypes
import numpy as np
import cv2

libc = ctypes.CDLL('../X64/Debug/trt_model1.dll')
#libc = ctypes.CDLL('../X64/Release/trt_model1.dll')

class trt_model(object):
    def __init__(self, val): 
        libc.trt_create_model.argtypes = [ctypes.c_char_p]
        libc.trt_create_model.restype = ctypes.c_void_p 
        self.obj = libc.trt_create_model(val.encode("utf8")) 
        
        U8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='A, C')
        libc.trt_input_data.argtypes = [ctypes.c_void_p, U8]
        libc.trt_input_data.restype = None

        libc.trt_run_model.restype = ctypes.c_void_p 
        libc.trt_run_model.restype = None

        F32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='A, C')
        libc.trt_output_data.argtypes = [ctypes.c_void_p, F32]
        libc.trt_output_data.restype = None

        libc.trt_clean_resouce.argtypes = [ctypes.c_void_p]
        libc.trt_clean_resouce.restype = None

    def trt_input_data(self, input_data):
        libc.trt_input_data(self.obj, input_data)

    def trt_run_model(self):
        libc.trt_run_model(ctypes.c_ulonglong(self.obj))

    def trt_output_data(self, output_data):
        libc.trt_output_data(self.obj, output_data)

    def trt_clean_resouce(self):
        libc.trt_clean_resouce(self.obj)

# COCO classes
coco_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def main():
    trt_m = trt_model("pythom demo!!! \n")
    img = cv2.imread('../etc/data/000000039769.jpg')  # image file load
    input_data = cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_LINEAR)
    input_data_re = np.reshape(input_data, (-1))
    output_data = np.zeros((100 * 95), dtype=np.float32)

    trt_m.trt_input_data(input_data_re)
    trt_m.trt_run_model()
    trt_m.trt_output_data(output_data)

    output_data_re = np.reshape(output_data, (100, -1))
    output_split = np.split(output_data_re, (91,95), axis=1)

    boxs = output_split[1]
    class_label = output_split[0].argmax(-1)
    probs = output_split[0].max(1)

    for idx, prob in enumerate(probs) :
        if prob > 0.9:
            label = class_label[idx]
            cx = boxs[idx][0]
            cy = boxs[idx][1]
            w = boxs[idx][2]
            h = boxs[idx][3]
            x1 = int((cx - w / 2.0) * img.shape[1])
            y1 = int((cy - h / 2.0) * img.shape[0])
            x2 = int((cx + w / 2.0) * img.shape[1])
            y2 = int((cy + h / 2.0) * img.shape[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
            label_text = "%s %.5f"%(coco_CLASSES[label],prob)
            cv2.putText(img, label_text, (x1, y1+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            print("%2d %2d %.5f %s"%(idx, label, prob, coco_CLASSES[label]))

    cv2.imshow("result", img)
    cv2.waitKey(0)

    trt_m.trt_clean_resouce()

if __name__ == "__main__":
	main()