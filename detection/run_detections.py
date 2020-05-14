import sys
sys.path.append('../src/')
import os
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
from ssd.models.keras_ssd512 import ssd_512
from ssd.keras_loss_function.keras_ssd_loss import SSDLoss
import tensorflow as tf
from retinanet.keras_retinanet import models
from retinanet.keras_retinanet import utils
from retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import cv2
import sys
from keras_yolo3.yolo import YOLO, detect_video
from PIL import Image
import argparse as ap

def get_yolo_boxes(yolo, image):
        out_boxes, out_scores, out_classes = yolo.detect_image(image)
        boxes = []
        scores = []
        classes = []
        for i in range(0, len(out_boxes)):       
            scores.append(out_scores[i])
            boxes.append(out_boxes[i])
            classes.append(out_classes[i].replace(" ", ""))
        return np.asarray(scores), np.asarray(boxes), classes
        
def get_retinanet_model(model_path):
        # load retinanet model
        model = models.load_model(model_path, backbone_name='resnet50')
        return model

def get_retinanet_results(image, model):
        # load label to names mapping for visualization purposes
        labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'trafficlight', 10: 'firehydrant', 11: 'stopsign', 12: 'parkingmeter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseballglove', 36: 'skateboard', 37: 'surfboard', 38: 'tennisracket', 39: 'bottle', 40: 'wineglass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hotdog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'pottedplant', 59: 'bed', 60: 'diningtable', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cellphone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddybear', 78: 'hairdrier', 79: 'toothbrush'}
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        # correct for image scale
        boxes /= scale
        boxes_people = []
        scores_people = []
        classes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.3: #or label != 0:
                break
                        
            b = box.astype(int)
            boxes_people.append([b[1],b[0],b[3],b[2]])
            scores_people.append(score)
            classes.append(labels_to_names[label].replace(" ", ""))
        return np.asarray(scores_people), np.asarray(boxes_people), classes
        
def ssd_model(weights_path, ssd__graph, ssd__session):
        img_height = 512
        img_width = 512
        with ssd__graph.as_default():
            with ssd__session.as_default(): 
                model = ssd_512(image_size=(img_height, img_width, 3),
                                n_classes=20,
                                mode='inference',
                                l2_regularization=0.0005,
                                scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                        [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                        [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                        [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                        [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                        [1.0, 2.0, 0.5],
                                                        [1.0, 2.0, 0.5]],
                                two_boxes_for_ar1=True,
                                steps=[8, 16, 32, 64, 128, 256, 512],
                                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                clip_boxes=False,
                                variances=[0.1, 0.1, 0.2, 0.2],
                                normalize_coords=True,
                                subtract_mean=[123, 117, 104],
                                swap_channels=[2, 1, 0],
                                confidence_thresh=0.5,
                                iou_threshold=0.45,
                                top_k=200,
                                nms_max_output_size=400)

                model.load_weights(weights_path, by_name=True)
                #Compile the model so that Keras won't complain the next time you load it.
                adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
                ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
                model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        return model

def get_ssd_results(image, model, ssd__graph, ssd__session):

        img_height = 512
        img_width = 512
                                                                                                   
        orig_images = [] # Store the images here.
        input_images = [] # Store resized versions of the images here.
        orig_images.append(image)
        img = cv2.resize(image, (img_height, img_width))
        img = np.asarray(img)
        input_images.append(img)
        input_images = np.array(input_images)
        with ssd__graph.as_default():   
            with ssd__session.as_default(): 
                y_pred = model.predict(input_images)
        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
        np.set_printoptions(precision=2, suppress=True, linewidth=90)        
        classes = ['background',
                           'airplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor']

        results = [];
        scores = [];
        classes_pred = [];
        for box in y_pred_thresh[0]:                
            # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
            ymin = box[-4] * orig_images[0].shape[1] / img_width  
            xmin = box[-3] * orig_images[0].shape[0] / img_height
            ymax = box[-2] * orig_images[0].shape[1] / img_width
            xmax = box[-1] * orig_images[0].shape[0] / img_height                        
            scores.append(box[1])
            results.append([xmin, ymin, xmax, ymax])
            classes_pred.append(classes[int(box[0])])
        return np.asarray(scores), np.asarray(results), classes_pred


def run(read_direc, save_direc, sessions_info, save_image=False):

    yolo_graph, yolo_session = sessions_info['yolo']
    ssd_graph, ssd_session = sessions_info['ssd']
    retinanet_graph, retinanet_session = sessions_info['retinanet']

    if os.path.exists('./processed_images.txt'):    
        with open('./processed_images.txt', 'r') as f:
            processed_files = f.readlines()
    else:
        processed_files = []

    if not os.path.exists(save_direc+'/'):
        os.mkdir(save_direc+'/')
    if save_image:  
        if not os.path.exists(save_direc+'/images/'):
            os.mkdir(save_direc+'/images/')
    
    i=0
    for fi in os.listdir(read_direc):
        if fi + '\n' in processed_files:
            print('Skipping ', fi)
            continue
        if fi == 'processed' or fi == 'processed-histeq-HSV' or fi == 'processed-all-classes' or fi == 'processed-all-classes':
            continue
                        
        image = cv2.imread(read_direc+fi)
            
        #histogram equalization            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:,:,2] = cv2.equalizeHist(image[:,:,2])
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
                
        i+=1
        if i % 1000 == 0:
            print('Processed ' + str(i) + 'images')
                        
        #Run dedection for each method
        with ssd_graph.as_default():
            with ssd_session.as_default():
                ssd_scores, ssd_boxes, ssd_classes = get_ssd_results(image, ssd_model, ssd_graph, ssd_session)
                            
        with yolo_graph.as_default():
            with yolo_session.as_default():         
                image2 = Image.fromarray(np.asarray(image))
                yolo_scores, yolo_boxes, yolo_classes = get_yolo_boxes(yolo, image2)  
                            
        with retinanet_graph.as_default():
            with retinanet_session.as_default():
                image3 = np.asarray(image2.convert('RGB'))
                image3 = image3[:, :, ::-1].copy()
                retinanet_scores, retinanet_boxes, retinanet_classes = get_retinanet_results(image3, retinanet)
               

        image = np.asarray(image)
        font = cv2.FONT_HERSHEY_SIMPLEX
                
        #Write results for each method
        for bbox, score, class_pred in zip(yolo_boxes, yolo_scores, yolo_classes):
            with open(save_direc+'/groundtruth_boxes_yolo.txt', 'a') as f:        
                f.write(str(fi) + ' ' + str(bbox[1])+ ' ' + str(bbox[0]) + ' ' + str(bbox[3]) + ' ' + str(bbox[2]) + ' ' + str(score) + ' ' + str(class_pred) + '\n')
            if save_image:     
                cv2.rectangle(image, (int(bbox[1]+1), int(bbox[0]+1)), (int(bbox[3]+1), int(bbox[2]+1)), (0,255,0), 3)
                cv2.putText(image,class_pred,(int(bbox[1]+1),int(bbox[0]+1)), font, 4,(0,255,0),2,cv2.LINE_AA)

        for bbox, score, class_pred in zip(ssd_boxes, ssd_scores, ssd_classes):         
            with open(save_direc+'/groundtruth_boxes_ssd.txt', 'a') as f:
                f.write(str(fi) + ' ' + str(bbox[1])+ ' ' + str(bbox[0]) + ' ' + str(bbox[3]) + ' ' + str(bbox[2]) + ' ' + str(score) + ' ' + str(class_pred) + '\n')                           
            if save_image:
                cv2.rectangle(image, (int(bbox[1]+0.5), int(bbox[0]+0.5)), (int(bbox[3]+0.5), int(bbox[2]+0.5)), (255,0,0), 3)
                cv2.putText(image,class_pred,(int(bbox[1]+1),int(bbox[0]+1)), font, 4,(255,0,0),2,cv2.LINE_AA)
                        
        for bbox, score, class_pred in zip(retinanet_boxes, retinanet_scores, retinanet_classes):
            with open(save_direc+'/groundtruth_boxes_retinanet.txt', 'a') as f:
                f.write(str(fi) + ' ' + str(bbox[1])+ ' ' + str(bbox[0]) + ' ' + str(bbox[3]) + ' ' + str(bbox[2]) + ' ' + str(score) + ' ' + str(class_pred) + '\n')
            if save_image:
                cv2.rectangle(image, (int(bbox[1]+1.5), int(bbox[0]+1.5)), (int(bbox[3]+1.5), int(bbox[2]+1.5)), (0,255,255), 3)
                cv2.putText(image,class_pred,(int(bbox[1]+1),int(bbox[0]+1)), font, 4,(0,255,255),2,cv2.LINE_AA)

        with open('./processed_images.txt', 'a') as f:
            f.write(fi + '\n')

        if save_image:
            cv2.imwrite(save_direc + '/images/' + fi + '-processed.jpg', image)
                                
                                
if __name__ == '__main__':

    parser = ap.ArgumentParser()
    #group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('-r', "--readdir", help="Directory with images")
    parser.add_argument('-s', "--savedir", help="Directory for saving the detection results")
    parser.add_argument('-i', "--saveimage", action='store_true', help="Save image with predicted bounding box or not")
    args = vars(parser.parse_args())
        
    read_direc = args['readdir'] 
    save_direc = args['savedir'] 
        
    ssd_weights_path = '../src/models/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5'
    retina_model_path = '../src/models/resnet50_coco_best_v2.0.3.h5'
    yolo_model_path = '../src/models/yolo.h5'		#"model_path": '../models/yolo.h5',
    #"anchors_path": '../keras_yolo3/model_data/yolo_anchors.txt',
    #"classes_path": '../keras_yolo3/model_data/coco_classes.txt',
    ssd_graph = tf.Graph()
    with ssd_graph.as_default():
        ssd_session = tf.Session()
        with ssd_session.as_default():  
            ssd_model = ssd_model(ssd_weights_path, ssd_graph, ssd_session)
                                        
    yolo_graph = tf.Graph()
    with yolo_graph.as_default():
        yolo_session = tf.Session()
        with yolo_session.as_default():
            yolo = YOLO(model_path=yolo_model_path)
                                        
    retinanet_graph = tf.Graph()
    with retinanet_graph.as_default():
        retinanet_session = tf.Session()
        with retinanet_session.as_default():                       
            retinanet = get_retinanet_model(retina_model_path)
                                        
    sessions_info = {'yolo': (yolo_graph, yolo_session), 'ssd': (ssd_graph, ssd_session), 'retinanet': (retinanet_graph, retinanet_session)}
    run(read_direc, save_direc, sessions_info, args['saveimage'])
    print('Finished')
               
