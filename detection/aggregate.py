import numpy as np
from utilsww import non_max_suppression, mean_boxes
from collections import defaultdict
import cv2
import os
import shutil
import pickle 
import json
import argparse as ap

def read_annotations(annot_dir):
    '''
    Input: annot_dir = directory with annotations
    Output: id2label = dictionary where key = photo id, value = photographer name
    '''
    id2labels = {}
        
    for fileannot in os.listdir(annot_dir):
        ph = fileannot.split('.txt')[0]
        with open(annot_dir + fileannot, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fileid = line.split('.')[0]
                fileid = fileid.split('/')[-1]
                id2labels[fileid] = ph
    return id2labels

def run(save_dir, load_gt_dir, load_img_dir, photographer_names, id2labels, save_image = False):

    #Create a placeholder for results
    results = {}
    for photographer in photographer_names:
        #results[photographer] = [{'close':[],'medium':[],'far':[]},{'person':[0,[]]}]
        results[photographer] = {'dist':{'close':[],'medium':[],'far':[]},'obj':{'person':[0,[]]},'photo_freq':0}

    save_img_dir = save_dir + '/images-aggregated/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if save_image and not os.path.exists(save_img_dir): 
        os.mkdir(save_img_dir)

    #Read the detections and aggregate them into dictionaries {file: [top, left, right, bottom, score, label]}
    ssd_boxes = np.loadtxt(load_gt_dir+'/groundtruth_boxes_ssd.txt', str)
    yolo_boxes = np.loadtxt(load_gt_dir+'/groundtruth_boxes_yolo.txt', str)
    retinanet_boxes = np.loadtxt(load_gt_dir+'/groundtruth_boxes_retinanet.txt', str)
    mask_boxes = np.loadtxt(load_gt_dir+'/groundtruth_boxes_mask.txt', str)

    total_boxes_d = defaultdict(list)
    for f,a,b,c,d,s,l in ssd_boxes:
        total_boxes_d[f].append(np.asarray([a,b,c,d,s,l]))
    for f,a,b,c,d,s,l in yolo_boxes:
        total_boxes_d[f].append(np.asarray([a,b,c,d,s,l]))
    for f,a,b,c,d,s,l in retinanet_boxes:
        total_boxes_d[f].append(np.asarray([a,b,c,d,s,l]))
    for f,a,b,c,d,s,l in mask_boxes:
        total_boxes_d[f].append(np.asarray([a,b,c,d,s,l]))

    i=0
    #iterate through images in load_img_dir and aggregate info for each of them
    for fi in os.listdir(load_img_dir):        
        photographer_name = id2labels[fi.split('.')[0]]

        #rename the photographers containing non-UTF characters
        if photographer_name[0] == 'J':
            photographer_name = 'Janis, P.'
        elif photographer_name.startswith('Heden'):
            photographer_name = 'Hedenstrom, O.'
        elif photographer_name.startswith('Sundst'):
            photographer_name = 'Sundstrom, H.'
        elif photographer_name.startswith('Sj'):
            photographer_name = 'Sjoblom, K.'

        if photographer_name not in photographer_names:
            continue

        results[photographer_name]['photo_freq'] +=1

        with open(save_dir+'/meta.txt', 'a') as f:
            f.write('\n________________________________________________________________________\n')
            f.write(fi+' \n')
            f.write('Photographer: ' + photographer_name + '\n')

        #read the image
        image = cv2.imread(load_img_dir+'/'+fi)
        image_area = np.shape(image)[0]*np.shape(image)[1]
        i+=1
        if i % 1000 == 0:
            print('Processed ' + str(i) + 'images')
                
        height = 800
        width = 800            
    
        #iterate through files for which there are detections
        if fi in total_boxes_d.keys():            
            close_objects = 0
            very_close_objects = 0
            portrait = 0
            obj_freq = {}
            num_people = 0
            standing_people = 0

            boxes = total_boxes_d[str(fi)]
            boxes = np.asarray(boxes) 

            classes_boxes = {}
            total_boxes = []
            total_scores=[]
            total_classes = []
            for x in boxes: 
                total_boxes.append(x[:-2])
                total_scores.append(x[-2])
                total_classes.append(x[-1])                    
                if x[-1] in classes_boxes.keys():
                    classes_boxes[x[-1]][0].append(x[:-2])
                    classes_boxes[x[-1]][1].append(x[-2])
                else:
                    classes_boxes[x[-1]] = [[],[]]
                    classes_boxes[x[-1]][0].append(x[:-2])
                    classes_boxes[x[-1]][1].append(x[-2])
            
            total_boxes = np.asarray(total_boxes)
            total_scores = np.asarray(total_scores)
            
            #we need to perform nms for each class separately. create dict class:[[total_boxes],[total_scores]]
            for k in classes_boxes.keys():

                classes_boxes[k][0] = np.asarray(classes_boxes[k][0])
                classes_boxes[k][1] = np.asarray(classes_boxes[k][1])
                        
                boxes_to_select_mean = mean_boxes(classes_boxes[k][0], classes_boxes[k][1], 0.1)    
                                
                indices = non_max_suppression(classes_boxes[k][0], classes_boxes[k][1], 0.1)
                boxes_to_select_nms = classes_boxes[k][0][indices]
                scores_to_select_nms = classes_boxes[k][1][indices]
                
                if k == 'person':
                    num_people = len(boxes_to_select_mean)
                    with open(save_dir+'/meta.txt', 'a') as f:
                        f.write('Estimated ' + str(num_people) + ' people\n')

                obj_freq[k] = len(boxes_to_select_mean)
                
                #save the results
                for bbox in boxes_to_select_mean:
                    with open(save_dir+'/groundtruth_boxes_mean.txt', 'a') as f:
                        f.write(fi + ' ' + str(bbox[1])+ ' ' + str(bbox[0]) + ' ' + str(bbox[3]) + ' ' + str(bbox[2]) + ' ' + str(bbox[4]) + ' ' + k + '\n')
                    
                for bbox, score in zip(boxes_to_select_nms, scores_to_select_nms):
                    with open(save_dir+'/groundtruth_boxes_nms.txt', 'a') as f:
                        f.write(fi + ' ' + str(bbox[1])+ ' ' + str(bbox[0]) + ' ' + str(bbox[3]) + ' ' + str(bbox[2]) + ' ' + str(score) + ' ' + k + '\n')
 
                for box in boxes_to_select_mean:    
                    if save_image:
                        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 10)
                        cv2.putText(image,k,(round(float(box[0])), round(float(box[1]))), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255),10,cv2.LINE_AA)                   

                    #estimate if it is a close-up or a cloase photo
                    if k=='person':
                        if (int(box[2]) - int(box[0]))*(int(box[3]) - int(box[1])) > 0.65*image_area:
                            very_close_objects += 1
                        elif (int(box[2]) - int(box[0]))*(int(box[3]) - int(box[1])) > 0.1*image_area:
                            close_objects += 1
                        if (int(box[3]) - int(box[1]))/(int(box[2]) - int(box[0])) > 2.1:
                            standing_people += 1

                
            with open(save_dir+'/meta.txt', 'a') as f:     
                if very_close_objects > 0:                         
                    f.write('Close photo\n')
                    results[photographer_name]['dist']['close'].append(fi)
                elif close_objects > 0:
                    f.write('Photo from medium distance\n')
                    results[photographer_name]['dist']['medium'].append(fi)
                else:
                    f.write('Photo from distance\n')
                    results[photographer_name]['dist']['far'].append(fi)
                if portrait:
                    f.write('Portrait\n')
                if num_people:
                    results[photographer_name]['obj']['person'][1].append((fi,obj_freq['person']))
                    results[photographer_name]['obj']['person'][0] += obj_freq['person']
                    obj_freq.pop('person')
                if standing_people:
                    f.write('People standing\n')
                if len(obj_freq.keys()):
                    f.write('Other objects: '+str(obj_freq)+'\n')
                    for obj_cl in obj_freq.keys():
                        if obj_cl in results[photographer_name]['obj'].keys():
                            results[photographer_name]['obj'][obj_cl][1].append((fi,obj_freq[obj_cl]))
                            results[photographer_name]['obj'][obj_cl][0] += obj_freq[obj_cl]
                        else:
                            results[photographer_name]['obj'][obj_cl] = [0,[]]
                            results[photographer_name]['obj'][obj_cl][1].append((fi,obj_freq[obj_cl]))
                            results[photographer_name]['obj'][obj_cl][0] += obj_freq[obj_cl]
            
                    f.write('________________________________________________________________________\n')

            if save_image:
               image = cv2.resize(image, (width, height))
               cv2.imwrite(save_img_dir+fi+'-proccessed.jpg', image)

    #save detailed results as json file
    with open(save_dir + '/photographer_results_dict.json', 'w') as f:
        json.dump(results, f) 
           
    for photographer in photographer_names:
        with open(save_dir + '/results.txt', 'a') as f:
            f.write(photographer + '\n')
            f.write('Total: ' + str(results[photographer]['photo_freq']) + '\n')
            f.write('close: ' + str(len(results[photographer]['dist']['close']))+ '\n')    
            f.write('medium: ' + str(len(results[photographer]['dist']['medium']))+ '\n') 
            f.write('far: ' + str(len(results[photographer]['dist']['far']))+ '\n') 
            for obj in results[photographer]['obj'].keys():
                f.write(obj + ': ' + str(results[photographer]['obj'][obj][0]) + ', ')  
            f.write('\n') 
            f.write('\n') 
                        
if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-r', "--readdir", help="Directory with images")
    parser.add_argument('-s', "--savedir", help="Directory for saving the detection results")
    parser.add_argument('-i', "--saveimage", action='store_true', help="Save image with predicted bounding box or not")
    args = vars(parser.parse_args())
        
    annot_dir = '../annotations/'
    save_dir = args['savedir'] #directory for the results
    load_gt_dir = save_dir.split('/aggregated')[0] #directory with output of detectors
    load_img_dir = args['readdir'] #original images
    photographer_names = ['Borg, K.', 'Ovaskainen, U.', 'Nousiainen', 'Hollming, V.','Taube, J.', 'Helander, N.', 'Janis, P.', 'Hedenstrom, O.', 'Suomela, E.', 'Norjavirta, T.', 'Persson, M.', 'Kivi, K.', 'Sundstrom, H.', 'Uomala, V.', 'Nurmi, E.', 'Harrivirta, H.', 'Aavikko, O.', 'Laukka, U.', 'Sjoblom, K.', 'Kyytinen, P.','Manninen, E.',
'Roivainen, H.','Kartto, T.']
    id2labels = read_annotations(annot_dir)
    run(save_dir, load_gt_dir, load_img_dir, photographer_names, id2labels, args['saveimage'])
    print('Finished')



