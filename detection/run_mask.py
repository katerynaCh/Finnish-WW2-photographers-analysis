import sys
sys.path.append('../src/')
import os
import numpy as np
from mask_rcnn.mrcnn import utils
import mask_rcnn.mrcnn.model as modellib
from mask_rcnn.samples.coco import coco
import cv2
import argparse as ap

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
        
        
def get_mask_rcnn(model, image, COCO_MODEL_PATH):
    # Run detection
    results = model.detect([image], verbose=1)

    r = results[0]
    idx = np.where(r['class_ids'] != 0) #select non-background
                
    boxes = r['rois'][idx]        
    scores = r['scores'][idx]
    classes = r['class_ids'][idx]  
                
    #score threshold = 0.3              
    idxs = np.where(scores > 0.3)
    boxes = boxes[idxs]
    people_scores = scores[idxs]
    classes = classes[idxs]
                
    return boxes, scores, classes
        
                
def run(read_direc, save_direc, model, COCO_MODEL_PATH, class_names, save_image=False):

    if os.path.exists('./processed_images_mask.txt'):    
        with open('./processed_images_mask.txt', 'r') as f:
            processed_files = f.readlines()
    else:
        processed_files = []
                
    print('Started:', save_direc, read_direc)
        
    if not os.path.exists(save_direc+'/'):
        os.mkdir(save_direc+'/')
    if save_image:
        if not os.path.exists(save_direc+'/images_mask/'):
            os.mkdir(save_direc + '/images_mask/')
    i=0
    for fi in os.listdir(read_direc):
           
        if fi + '\n' in processed_files:
            print('Skipping ', fi)
            continue
                        
        image = cv2.imread(read_direc +fi)
                
        #histogram equalization
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:,:,2] = cv2.equalizeHist(image[:,:,2])
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        i = i+1
        if i % 1000 == 0:
            print('Processed ' + str(i) + 'images')
        scaler_y = np.shape(image)[0]/960
        scaler_x = np.shape(image)[1]/540
        image1 = cv2.resize(image, (540, 960))
                
        mask_boxes, mask_scores, mask_classes = get_mask_rcnn(model, image1, COCO_MODEL_PATH)
           
        for bbox, score, classid in zip(mask_boxes, mask_scores, mask_classes):
            bbox[1] = int(bbox[1])*scaler_x
            bbox[0] = int(bbox[0])*scaler_y
            bbox[3] = int(bbox[3])*scaler_x
            bbox[2] = int(bbox[2])*scaler_y 
            with open(save_direc+'/groundtruth_boxes_mask.txt', 'a') as f:        
                f.write(str(fi) + ' ' + str(bbox[1])+ ' ' + str(bbox[0]) + ' ' + str(bbox[3]) + ' ' + str(bbox[2]) + ' ' + str(score) + ' ' + class_names[classid] + '\n')
            if save_image:     
                cv2.rectangle(image, (int(bbox[1]+1), int(bbox[0]+1)), (int(bbox[3]+1), int(bbox[2]+1)), (0,255,0), 3)
                cv2.putText(image, class_names[classid], (round(float(bbox[1])), round(float(bbox[0]))), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255),10,cv2.LINE_AA)        
                
            with open('./processed_images_mask.txt', 'a') as f:
                f.write(fi + '\n')
                        
            if save_image:
                cv2.imwrite(save_direc+'/images_mask/' + str(i) + '.jpg', image)

                        
if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-r', "--readdir", help="Directory with images")
    parser.add_argument('-s', "--savedir", help="Directory for saving the detection results")
    parser.add_argument('-i', "--saveimage", action='store_true', help="Save image with predicted bounding box or not")
    args = vars(parser.parse_args())
        
    read_direc = args['readdir'] 
    save_direc = args['savedir'] 
        
    COCO_MODEL_PATH = "../src/models/mask_rcnn_coco.h5"
    MODEL_DIR = os.path.join('mask_rcnn/', "logs") 
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                                   'bus', 'train', 'truck', 'boat', 'trafficlight',
                                   'fire hydrant', 'stop sign', 'parkingmeter', 'bench', 'bird',
                                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball',
                                   'kite', 'baseballbat', 'baseballglove', 'skateboard',
                                   'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup',
                                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                                   'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza',
                                   'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed',
                                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                                   'keyboard', 'cellphone', 'microwave', 'oven', 'toaster',
                                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                                   'teddybear', 'hairdrier', 'toothbrush']
    config = InferenceConfig()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    run(read_direc, save_direc, model, COCO_MODEL_PATH, class_names, args['saveimage'])
    print('Finished')
                           
 
