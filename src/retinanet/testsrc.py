
# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet import backend
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	# adjust this to point to your downloaded/trained model
	# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('snapshots', 'resnet50_coco_best_v2.0.3.h5')
image = read_image_bgr('./testimage2.jpg')

def get_retinanet_model(model_path):


	# load retinanet model
	model = models.load_model(model_path, backbone_name='resnet50')
	return model

def get_retinanet_results(image, model):
	# load label to names mapping for visualization purposes
	labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

	
	# preprocess image for network
	image = preprocess_image(image)
	image, scale = resize_image(image)

	boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

	# correct for image scale
	boxes /= scale
	boxes_people = []
	scores_people = []
	for box, score, label in zip(boxes[0], scores[0], labels[0]):
		# scores are sorted so we can break
		if score < 0.5 or label != 0:
			break
			
		b = box.astype(int)
		boxes_people.append(b)
		scores_people.append(score)
	return np.asarray(boxes_people)
	
retinanet = get_retinanet_model(model_path)
print(get_retinanet_results(image, retinanet))	