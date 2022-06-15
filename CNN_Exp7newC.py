# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 19:03:40 2020
Modified on Fri Apr 01 11:19:00 2022
@author: Andres

Training TCGA dataset

"""

Exp = 'Exp7'
Exp_path = '/home/asg143/experimento7/'

import tensorflow as tf
import tensorflow.keras as keras
import math
import albumentations as A

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
        #A.Resize(112, 112, interpolation=1, always_apply=True, p=1),
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

#%%

train_path = '/home/usuario/Descargas/destino 20/'
validation_path = '/home/usuario/Descargas/destino 20/'
test_path = '/home/usuario/Descargas/destino 20/'

modelpath = '/home/usuario/Documentos/GBM/TCGA/'
modelname = 'best_model22102021_ResNet50Exp8.h5'

#%%

batch_size = 32
imwidth,imheight = (224,224)
#target_size = imwidth,imheight 
class_mode = 'categorical'
classes = ['NE','CT']

TrainDatagen = ImageDataGenerator(preprocessing_function=transform,
                                  validation_split=0.3)
ValDatagen = ImageDataGenerator(rescale=1./255,
                                validation_split=0.3)
TestDatagen = ImageDataGenerator(rescale=1./255)

TrainingData = TrainDatagen.flow_from_directory(train_path,
                                                  target_size=(imwidth,imheight), 
                                                  classes=classes,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  seed=1,
                                                  subset='training',
                                                  class_mode=class_mode)

ValidationData = ValDatagen.flow_from_directory(validation_path,
                                                    target_size=(imwidth,imheight),
                                                    classes=classes,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    subset='validation',
                                                    class_mode='categorical')

TestData = ValDatagen.flow_from_directory(test_path,
                                            target_size=(imwidth,imheight), 
                                            classes=classes,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            class_mode='categorical')

#%% Define model ####

model = Sequential()
"""
model.add(tf.keras.applications.EfficientNetB0(
          include_top=True,
          weights=None,
          input_tensor=None,
          input_shape=(imwidth,imheight,3),
          pooling=None,
          classes=2,
          classifier_activation="softmax"))
"""

model.add(tf.keras.applications.ResNet50(include_top=True,
                                         weights=None,
                                         input_tensor=None,
                                         input_shape=(224, 224, 3),
                                         pooling=None,
                                         classes=2,))
    
model.summary()

#%%

modelpath = '/home/usuario/Documentos/GBM/TCGA/'
modelname = 'best_model22102021_ResNet50Exp8.h5'

model = tf.keras.models.load_model('/home/usuario/Documentos/GBM/TCGA/best_model22102021_ResNet50Exp8.h5')

weights = model.get_weights()

#%%

from tensorflow.keras.layers import Input
from keras.utils.vis_utils import plot_model

#%%
model = Sequential()
"""
model.add(tf.keras.applications.EfficientNetB0(
          include_top=True,
          weights=None,
          input_tensor=None,
          input_shape=(imwidth,imheight,3),
          pooling=None,
          classes=2,
          classifier_activation="softmax"))
"""

model.add(tf.keras.applications.ResNet50(include_top=True,
                                         weights=None,
                                         input_tensor=None,
                                         input_shape=(224, 224, 3),
                                         pooling=None,
                                         classes=2,))
    
# model.summary()


input_layer = Input((224,224,3),name='Name Layer')

mdl = model.layers[0]
numlayers = len (mdl.layers)
print(numlayers)

SplitModel=Model(inputs=mdl.inputs,
                 outputs=mdl.layers[numlayers-2].output,
                 name='ResNet50')

input_img = Input(shape=(224,224,3),name='Input')
outSplitModel = SplitModel(input_img)
output1 = Dense(2,activation='softmax',name='OldOut')(outSplitModel)
output2 = Dense(2,activation='softmax',name='NewOut')(outSplitModel)

model3 = Model(input_img,outputs=[output1,output2])


plot_model(model3,to_file='/home/usuario/Descargas/eso.png')


#%%

modelpathTL = '/home/usuario/Documentos/GBM/TCGA/'
modelnameTL = 'TL_best_model22102021_ResNet50Exp8.h5'

#%% Compile model

import numpy as np
from tensorflow.python.ops import math_ops
CCE=tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
CCE2=tf.keras.losses.CategoricalCrossentropy()


#%%

from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras import backend as K

def oldloss(y_true, y_pred):
    y_pred_hat = tf.sqrt(y_pred)
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(y_true,y_pred_hat)
    return loss

def newloss(y_true, y_pred):
    # y_pred_hat = tf.sqrt(y_pred)
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(y_true,y_pred)
    return loss

def LwFloss(y_true, y_pred):
    y_pred_hat = tf.sqrt(y_pred)
    
    cce = tf.keras.losses.CategoricalCrossentropy()
    
    CCEold = cce(y_true,y_pred_hat)
    CCEnew = cce(y_true,y_pred)
    
    wCCEold = tf.math.scalar_mul(0.1, CCEold, name=None)
    wCCEnew = tf.math.scalar_mul(0.9, CCEnew, name=None)

    loss = tf.math.add(wCCEold, wCCEnew, name=None)
    return loss


#%%


y_true = [[0,1], [0,1]]
y_pred = [[0.05, 0.95], [0.2, 0.8]]

# y_pred2 = np.array(y_pred)**(1/2)
# y_pred2 = tf.sqrt(y_pred)
ol = oldloss(y_true,y_pred).numpy()
nl = newloss(y_true,y_pred).numpy()
lwfl = LwFloss(y_true,y_pred).numpy()
print('old_loss:' + str(0.1*ol))
print('new_loss:' + str(0.9*nl))


print('lwf_loss:' + str(lwfl))




# cce = tf.keras.losses.CategoricalCrossentropy()

# zz = cce(y_true,y_pred)
# zz2 = cce(y_true,y_pred2)
# x = tf.math.scalar_mul(0.1, zz2, name=None)
# y = tf.math.scalar_mul(0.9, zz, name=None)

# ll = tf.math.add(x, y, name=None)


#%%

def step_decay(epoch):
	initial_lrate = 1e-5
	drop = 0.1
	epochs_drop = 10
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

alpha=0.1
model3.compile(optimizer=Adam(1e-5), 
                loss = {'NewOut':newloss,'OldOut':oldloss},
                loss_weights={'OldOut': alpha,
                              'NewOut': 1-alpha },
                metrics=['accuracy'])
# alpha=0.1
# model3.compile(optimizer=Adam(1e-5), 
#                 loss = [oldloss,newloss],
#                 # loss_weights=[1,alpha,1-alpha],
#                 # loss_weights={'OldOut': 0.1,
#                 #               'NewOut': 1 },
#                 metrics=['accuracy'])



# model3.compile(
#                 optimizer=Adam(1e-5), 
#                 loss = alpha_loss,
#                 metrics=['accuracy'])

lr = LearningRateScheduler(step_decay)

es = EarlyStopping(patience = 10,
                   mode = 'min', 
                   verbose = 1)

mc = ModelCheckpoint(filepath = modelpathTL + modelnameTL, 
                     monitor = 'val_loss', 
                     save_weights_only=True,
                     verbose = 1, 
                     save_best_only = True, 
                     mode = 'min', 
                     save_freq = 'epoch')

#%% Training and Validation #########

epochs = 100

TrainSteps = TrainingData.samples // batch_size
ValidSteps = ValidationData.samples // batch_size
TestSteps  = TestData.samples // batch_size

# modelpathTL = '/home/usuario/Documentos/GBM/TCGA/'
# modelnameTL = 'best_model22102021_ResNet50Exp8.h5'

# for ind in range(0,5):

# model3 = models.clone_model(model2)

# model3.load_weights= modelpathTL + modelnameTL

# model3 = unfreezemodel(model3,ind)

# model3.summary()

    
# model3.compile(optimizer=Adam(1e-5), 
#               loss = 'categorical_crossentropy',
#               metrics=['accuracy'])
    
history = model3.fit(TrainingData,
                    steps_per_epoch  = TrainSteps,
                    validation_data  = ValidationData,
                    validation_steps = ValidSteps,
                    epochs = epochs,
                    verbose = 1,
                    callbacks = [es,mc]
                    # callbacks = [es,mc,lr]                    
                    )

    # modelpathTL = '/home/usuario/Documentos/GBM/TCGA/'
    # modelnameTL = 'TL_best_model22102021_ResNet50Exp8.h5'
    
# print('Fin del entrenamiento')


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