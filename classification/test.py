import keras
import pickle 
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
import argparse as ap

def plot_confusion_matrix(y_true, y_pred):
    photographer_names2 = ['Hollming, V.', 'Taube, J.',
    'Helander, N.', 'Jänis, P.', 'Hedenström, O.', 'Suomela, E.', 'Kivi, K.', 'Uomala, V.', 'Nurmi, E.',
                                                                                                'Sjöblom, K.', 'Roivainen, H.', 'Manninen, E.']

    photographer_names = [4,5,6,7,8,9,12,14,15,19,21,22]
    photographer_names2 = [0,1,2,3,4,5,6,7,8,9,10,11]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, photographer_names2)
    cmap=plt.cm.Blues

    plt.figure(figsize = (10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.colorbar(shrink=0.75).ax.tick_params(labelsize=15)
    tick_marks = np.arange(len(photographer_names))
    plt.xticks(tick_marks, photographer_names, rotation=45, fontsize=15)
    plt.yticks(tick_marks, photographer_names, fontsize=15)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=15,
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=15, labelpad=14)
    plt.xlabel('Predicted label', fontsize=15, labelpad=14)
    plt.subplots_adjust(bottom=.25, left=.25)
    plt.show()
    plt.savefig('cm.png')
                
def test(model, labels, partition, image_dir):
    test_ids = partition['test']
    train_ids = partition['train']
    y_train = [labels[i] for i in train_ids]
    n_classes = len(np.unique(y_train))
        
    total_correct = 0
    cnt = 0
    y_true = [labels[i] for i in test_ids]
    y_pred = []
    correct_images = []
    wrong_images = []
    for descriptor in test_ids:
        print('Image ', cnt)
        prediction = labels[descriptor]    
        print(image_dir + descriptor + '.tif')
        image = cv2.imread(image_dir + str(descriptor) + '.tif')
        image = cv2.resize(image,(224,224))
        predict_proba = model.predict(np.expand_dims(image,axis=0))             
        predicted_class = np.argmax(predict_proba)
        
        if predicted_class == labels[descriptor]:
            correct_images.append(descriptor)
            total_correct += 1
        else:
            wrong_images.append(descriptor)                        
        y_pred.append(predicted_class)
        cnt = cnt + 1          

    print('Accuracy: ', total_correct/len(test_ids))
    with open('./result.txt', 'w') as f:
        f.write('Accuracy: ' + str(total_correct/len(test_ids)))
    return y_true, y_pred

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-r', "--readdir", help="Directory with images")
    args = vars(parser.parse_args())
    image_dir = args['readdir']
    model= keras.models.load_model('./models/model.h5')
        
    with open('./partition.pkl', 'rb') as f: 
        partition = pickle.load(f)

    with open('./id2labels.pkl', 'rb') as f:
        labels = pickle.load(f) 
                        
    y_true, y_pred = test(model, labels, partition, image_dir)
    plot_confusion_matrix(y_true, y_pred)
