#set these imports to ensure repeatable results
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Input
from keras.layers import Conv2D, Cropping2D, BatchNormalization, MaxPooling2D
from keras import initializers
from keras.optimizers import Adam

from keras import backend as K
import gc
#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas
import matplotlib.pyplot as plt
import generate_data as gd
from keras import regularizers
from keras.models import Model
import random
import itertools
import os
from matplotlib import pyplot as plt

os.system("rm -f model.h5")
model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 40), (0, 0))))

model.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2), activation='relu', name='Conv1'))
    #model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
model.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu', name='Conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu', name='Conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#model.add(BatchNormalization())
model.add(Conv2D(128, (2, 2), padding='same', strides=(1, 1), activation='relu', name='Conv4'))
#model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='linear', name='FC1'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='linear', name='FC2'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='linear', input_shape=(128,), name='FC3'))
model.add(Dense(16, activation='linear', input_shape=(64,), name='FC4'))
model.add(Dense(1, name='angle_out'))

model.summary()
#model = Model(inputs=[input_image], outputs=[angle_out, speed_out])
#model = Model(inputs=[input_image], outputs=[angle_out])

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(optimizer=adam, loss='mse')

model.compile(optimizer='adam', loss= {'angle_out': 'mean_squared_error'})
#              loss_weights={'angle_out': 0.5, 'speed_out': .5})
#model.compile(optimizer='adam',
#              loss={'angle_out': 'mean_squared_error',
#                    'speed_out': 'mean_squared_error'},
 #             loss_weights={'angle_out': 0.5, 'speed_out': .5})

#model.compile(loss='mse', optimizer='adam')
##end of model definition

##train

#folders = ['AlexT-1']
folders = ['AlexT-1', 'AlexT-2', 'AlexT-3', 'AlexT-4']
df = gd.assemble_data(folders)
df = gd.clean_dataframe(df)
df = df.sample(frac=1).reset_index(drop=True)

# train the network
initial_batch_size = 128
epochs = 3

#batch_size = 128
train_cost = []
run_number = 0
batch_size = 128
validation_batch_size = 100
steps_per_epoch = 256
train_iter = gd.gen_batch(df,batch_size=batch_size)

validation_iter = gd.gen_batch(df,batch_size=validation_batch_size)  
X_valid_images, y_valid_angles, y_valid_speeds = next(validation_iter)      
#epoch = 6
history = model.fit_generator(train_iter, steps_per_epoch=steps_per_epoch, 
                              epochs=epochs,
                              validation_data=(X_valid_images, y_valid_angles), 
                              verbose=1)

#fit_generator(<generator..., validation_data=(array([[[..., verbose=1, steps_per_epoch=248, epochs=2)`
#for epoch in range(epochs):
#   batch_size = initial_batch_size + epoch*64
#   batch_size = np.min([batch_size, 256])
#   print("Training model.  Epoch# ", epoch, " Batch size: ", batch_size)  
#   for batch in range(4000//batch_size):
#     run_number += 1
#     train_iter = gd.gen_batch(df,batch_size=batch_size)
#     X_train, y_train_angle, y_train_speed = next(train_iter)
     #Model.train_on_batch(model, X_train, [y_train_angle, y_train_speed])
#     train = model.train_on_batch(X_train, y_train_angle)
#     train_cost.append(train)
        
#     if run_number%20 == 0:
#        print("cost (last 10): ", np.mean(train_cost[-10:]),   
#              " batch: ", batch+1)
 
        #steering_angle = float(model.predict(X_valid, batch_size=128))
#        X_valid_images, y_valid_angles, y_valid_speeds = next(validation_iter)            
#        predicted_angles = model.predict(X_valid_images)
        #errors = np.square(y_valid_angles - steering_angles)
        
#        for truth_angle, predicted_angle in zip(y_valid_angles, predicted_angles):
#            print("Angle prediction:" , predicted_angle, 
#                  " Ground Truth: ", truth_angle,
#                  " Error: ", truth_angle-predicted_angle)
            
           
        #print("*****STEERING ANGLE MEAN: ", np.mean(steering_angle),
        #      " GROUND TRUTH: ", np.mean(y_valid_angle),
        #      "**********")
        
        #print("******STEERING ABS(ANGLE) MEAN: ", np.mean(np.abs(steering_angle)),
        #      " GROUND TRUTH: ", np.mean(np.abs(y_valid_angle)),
        #      "*******")             
        #print("*****MEAN PREDICTION ERROR (y_truth - predict): ",
        #      np.mean(prediction_error))
print("saving model...")
model.save('./model.h5')  

gc.collect()
K.clear_session()


