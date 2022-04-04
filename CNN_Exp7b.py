# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 19:03:40 2020
Modified on Fri Apr 01 11:19:00 2022
@author: Andres
"""

Exp = 'Exp7'
Exp_path = '/home/asg143/experimento7/'

import tensorflow as tf
import tensorflow.keras as keras
import math
import albumentations as A
import pandas as pd
import os

from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential,datasets, layers, models

from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.losses import categorical_crossentropy
from keras.models import Model
#from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout,BatchNormalization, Activation, Input
#from keras import Sequential,datasets, layers, models



from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50

tf.__version__
keras.__version__

#%%


def transform(image):
    transform = A.Compose([
        A.ToFloat(max_value = 255,always_apply=True,p=1.0),
        A.Resize(112, 112, interpolation=1, always_apply=True, p=1),
        #A.RGBShift(always_apply=False, p=0.2, r_shift_limit=(-0.1, 0.1), g_shift_limit=(-0.1, 0.1), b_shift_limit=(-0.1, 0.1)),   
        A.RandomContrast(always_apply=False, p=0.3, limit=(-0.3, 0.3)),
        #A.HueSaturationValue(always_apply=False, p=0.5, hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3)),
        #A.RandomGamma(always_apply=False, p=0.5, gamma_limit=(50, 150), eps=1e-07), 
        #A.Blur(always_apply=False, p=0.5, blur_limit=(2, 4)),
        A.RandomBrightnessContrast(always_apply=False, p=0.3, brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), brightness_by_max=False),  
        A.ChannelShuffle(always_apply=False, p=0.3),
        A.Rotate(always_apply=False, p=0.5, limit=(-90, 90), interpolation=4, border_mode=4, value=(0, 0, 0), mask_value=None),
        A.VerticalFlip(always_apply=False, p=0.5),
        A.HorizontalFlip(always_apply=False, p=0.5),
        #A.Downscale(always_apply=False, p=0.5, scale_min=0.5, scale_max=0.8999999761581421, interpolation=0),
    ])
    return transform(image=image)['image']

# class albumentations.augmentations.geometric.resize.Resize (height, width, interpolation=1, always_apply=False, p=1)


#%%

train_path = '/home/usuario/Documentos/GBM/Samples/SomeSamples/trainX/'
validation_path = '/home/usuario/Documentos/GBM/Samples/SomeSamples/valX/'
test_path = '/home/usuario/Documentos/GBM/Samples/SomeSamples/testX/'

batch_size = 4
imwidth,imheight = (112,112)
target_size = imwidth,imheight 
class_mode = 'categorical'
classes = ['NE','CT']

TrainDatagen = ImageDataGenerator(preprocessing_function=transform)
ValDatagen = ImageDataGenerator(rescale=1./255)
TestDatagen = ImageDataGenerator(rescale=1./255)

TrainingData = TrainDatagen.flow_from_directory(train_path,
                                                  target_size=target_size, 
                                                  classes=classes,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  seed=1,
                                                  class_mode=class_mode)

ValidationData = ValDatagen.flow_from_directory(validation_path,
                                                    target_size=target_size,
                                                    classes=classes,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    class_mode='categorical')

TestData = ValDatagen.flow_from_directory(test_path,
                                            target_size=target_size, 
                                            classes=classes,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            class_mode='categorical')

TrainSteps = TrainingData.samples // batch_size
ValidSteps = ValidationData.samples // batch_size
TestSteps  = TestData.samples // batch_size

#%% Define model ####

model = Sequential()

model.add(tf.keras.applications.EfficientNetB0(
          include_top=True,
          weights=None,
          input_tensor=None,
          input_shape=(imwidth,imheight,3),
          pooling=None,
          classes=2,
          classifier_activation="softmax"))
    
model.summary()

#%% Compile model

def step_decay(epoch):
	initial_lrate = 1e-4
	drop = 0.1
	epochs_drop = 15
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

optimizer = Adam(1e-4) # 1e-5
model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

modelpath = '/home/usuario/Documentos/GBM/'
modelname = 'EffNetB0_31032022.h5'


lr = LearningRateScheduler(step_decay)

es = EarlyStopping(patience=10,
                   mode='min', 
                   verbose=1)

mc = ModelCheckpoint(filepath = modelpath + modelname, 
                     monitor='val_loss', 
                     verbose=1, 
                     save_best_only=True, 
                     mode='min', 
                     save_freq='epoch')

#%% Training and Validation #########

epochs = 100

history = model.fit(TrainingData,
                    steps_per_epoch  = TrainSteps,
                    validation_data  = ValidationData,
                    validation_steps = ValidSteps,
                    epochs = epochs,
                    verbose = 2,
                    use_multiprocessing = True,
                    callbacks = [es,mc,lr])

# Save last model
#checkpoint_final_path = dir2 + 'best_model_lastXX' + Exp + '.h5'
#model.save(checkpoint_final_path) 

#%%
                                   
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1, step=0.05))
plt.grid(color='k', linestyle='--', linewidth=0.4)
plt.legend(loc='lower right')
plt.savefig(Exp_path + 'accuracy_CNN_ ' + Exp +  '.png')

plt.figure(2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0, 1])
#plt.yticks(np.arange(0, 1, step=0.05))
plt.grid(color='k', linestyle='--', linewidth=0.4)
plt.legend(loc='lower right')
plt.savefig(Exp_path + 'loss_CNN_' + Exp + '.png')

############### PREDICTION ###############################
"""
# Predict Best Model
model=keras.models.load_model(dir2 + 'best_model' + Exp + '.h5')
predict = model.predict(test_batches,steps =steps_test+1,verbose=0)
predict_eval = model.evaluate(test_batches,steps =steps_test+1,verbose=0)

print("************")
print("Evaluation metrics")
print(predict_eval)
print("************")

import sklearn as sklearn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

predicted_label=np.round(predict[:,1])
predicted_label=np.int32(predicted_label)
true_label=test_batches.classes


# Area under the Curve (AUC)
auc = roc_auc_score(true_label,predict[:,1])
fpr, tpr, thresholds = roc_curve(true_label,predict[:,1])
plt.figure(3)
plt.plot(fpr,tpr, label = "AUC = " + str(auc))
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(color='k', linestyle='--', linewidth=0.4)
plt.legend(loc='lower right')
plt.savefig(dir + 'AUC_CNN_ ' + Exp +  '.png')

cm = confusion_matrix(true_label,predicted_label)

tn, fp, fn, tp = confusion_matrix(np.float32(true_label),np.float32(predicted_label)).ravel()

print('***** PREDICTION ******')

print('tp: '+np.str(tp))
print('tn: '+np.str(tn))
print('fp: '+np.str(fp))
print('fn: '+np.str(fn))

print('***** PREDICTION METRICS ******')

acc=(tp+tn)/(tn+tp+fn+fp)
sens=tp/(tp+fn)
spec=tn/(tn+fp)

print('accuracy: ' + str(acc))
print('sens: ' + str(sens))
print('spec: ' + str(spec))
print('AUC: ' + str(auc))

print("****** The process has ended *******")
"""