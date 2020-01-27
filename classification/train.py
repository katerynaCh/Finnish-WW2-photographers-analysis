import os
os.environ['PYTHONHASHSEED']=str('42')
import random
random.seed(42)
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)
import keras
import pickle
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dropout
import numpy as np
from sklearn.utils import class_weight
from utils import DataGenerator
import argparse as ap

def get_model(n_classes, initialization='imagenet'):
    #initialize base network (VGG19)
    base_model = keras.applications.vgg19.VGG19(include_top=True, weights=initialization, input_tensor=None, input_shape=None, pooling=None, classes=1000)
    base_model.layers.pop()
    base_model.outputs = [base_model.layers[-1].output]

    #insert dropout and fully-connected layers
    bl1p = base_model.get_layer('block1_pool')
    bl2c1 = base_model.get_layer('block2_conv1')
    bl2c2 = base_model.get_layer('block2_conv2')
    bl2p = base_model.get_layer('block2_pool')
    bl3c1 = base_model.get_layer('block3_conv1')
    bl3c2 = base_model.get_layer('block3_conv2')
    bl3c3 = base_model.get_layer('block3_conv3')
    bl3c4 = base_model.get_layer('block3_conv4')
    bl3p = base_model.get_layer('block3_pool')
    bl4c1 = base_model.get_layer('block4_conv1')
    bl4c2 = base_model.get_layer('block4_conv2')
    bl4c3 = base_model.get_layer('block4_conv3')
    bl4c4 = base_model.get_layer('block4_conv4')
    bl4p = base_model.get_layer('block4_pool')
    bl5c1 = base_model.get_layer('block5_conv1')
    bl5c2 = base_model.get_layer('block5_conv2')
    bl5c3 = base_model.get_layer('block5_conv3')
    bl5c4 = base_model.get_layer('block5_conv4')
    bl5p = base_model.get_layer('block5_pool')
    fl = base_model.get_layer('flatten')
    dn1 = base_model.get_layer('fc1')
    dn2 = base_model.get_layer('fc2')
    x = Dropout(0.5)(bl1p.output)
    x = bl2c1(x)
    x = bl2c2(x)
    x = bl2p(x)
    x = Dropout(0.5)(x)
    x = bl3c1(x)
    x = bl3c2(x)
    x = bl3c3(x)
    x = bl3c4(x)
    x = bl3p(x)
    x = Dropout(0.5)(x)
    x = bl4c1(x)
    x = bl4c2(x)
    x = bl4c3(x)
    x = bl4c4(x)
    x = bl4p(x)
    x = Dropout(0.5)(x)
    x = bl5c1(x)
    x = bl5c2(x)
    x = bl5c3(x)
    x = bl5c4(x)
    x = bl5p(x)
    x = Dropout(0.5)(x)
    x = fl(x)
    x = dn1(x)
    x = Dropout(0.5)(x)
    x = dn2(x)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    output = Dense(output_dim=n_classes, activation='softmax')(x)
    new_model = Model(base_model.input, output)
    return new_model


def train(params, data_dir, partition, labels):

    model = get_model(params['n_classes'])
    model.summary()
        
    #define data generators for training and validation
    gen_params = {'dim': (224,224),
                  'batch_size': params['batch_size'],
                  'n_classes': params['n_classes'],
                  'n_channels': params['n_channels'],
                  'shuffle': params['shuffle']}

    training_generator = DataGenerator(data_dir, partition['train'], labels, **gen_params)
    validation_generator = DataGenerator(data_dir, partition['validation'], labels, **gen_params)
        
    #compute class weights              
    y_train = [labels[i] for i in partition['train']]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    #define optimizer and compile the model
    adam = Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    #define callbacks
    earlyStopping = EarlyStopping(monitor='val_acc', patience=15, verbose=0, mode='max')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_acc', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='max')
                   
    model.fit_generator(epochs=params['epochs'],
                            generator=training_generator,
                            validation_data=validation_generator, 
                            callbacks=[mcp_save,reduce_lr_loss,earlyStopping],
                            use_multiprocessing=True,
                            shuffle=True,
                            workers=2,
                            class_weight = class_weights)
    model.save('./models/model.h5')
        
if __name__ == '__main__':
    parser = ap.ArgumentParser()
        
    parser.add_argument('-r', "--readdir", help="Directory with images")
    args = vars(parser.parse_args())
    data_dir = args['readdir'] 

    with open('partition.pkl', 'rb') as f:
        partition = pickle.load(f)

    with open('id2labels.pkl', 'rb') as f:
        labels = pickle.load(f)
                        
    params = {'batch_size': 8,
             'n_classes': 12,
             'n_channels': 3,
             'shuffle': True, #shuffle data during training
             'lr': 1e-5, #learning rate
             'epochs': 100}
    train(params, data_dir, partition, labels)
