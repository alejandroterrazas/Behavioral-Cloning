model = Squential()
#input_image = Input(shape=(160,320, 3), name='input_image')
#x = input_image
#model.add(input_image)
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320,3)))
##crop 60 off the top and 40 off the bottom
x = Cropping2D(cropping=((60, 40), (0, 0)))(x)

x = Conv2D(64, (3, 3),
          kernel_initializer = 'glorot_uniform',
#          bias_initializer=initializers.Constant(0.1),
#          use_bias = True,
          strides=(2, 2), 
          activation='elu')(x)

x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
#x = BatchNormalization()(x)

x = Conv2D(128, (3, 3), 
           kernel_initializer = 'glorot_normal',
 #          bias_initializer=initializers.Constant(0.1),
 #          use_bias = True,
           padding = 'same',
           strides=(2, 2), 
           activation='elu')(x)

x = MaxPooling2D(pool_size=(2, 2), padding='same', data_format=None)(x)
#x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
x = Conv2D(256, (3, 3), kernel_initializer = 'glorot_normal',
#           bias_initializer=initializers.Constant(0.1),
#           use_bias = True,
           padding = 'same',
           strides=(2, 2), 
           activation='elu')(x)

x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
#x = BatchNormalization() (x)
#x = Dropout(0.5)(x)
x = Conv2D(128, (1, 1), 
           kernel_initializer = 'glorot_normal',
#           bias_initializer=initializers.Constant(0.1),
#           use_bias = True,
           padding = 'same',
           strides = (1,1),
           activation='elu')(x)
           
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
#x = BatchNormalization() (x)

x = Conv2D(64, (1, 1), 
           kernel_initializer = 'glorot_normal',
#           bias_initializer=initializers.Constant(0.1),
#           use_bias = True,
           padding = 'same',
           strides = (1,1),
           activation='elu')(x)
           
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
#x = Dropout(0.5)(x)
#x = BatchNormalization() (x)

##DONE WTIH CONVOLUTIONAL LAYERS 

x = Flatten()(x)

x = Dense(128,
          kernel_initializer = 'truncated_normal',         
          bias_initializer = 'zeros',
          use_bias = True,
          activation='linear',
          kernel_regularizer=regularizers.l2(0.01))(x)

x = Dropout(0.5)(x)

x = Dense(64,
          kernel_initializer = 'truncated_normal',
          bias_initializer='zeros', 
          use_bias = True,
          activation = 'linear',
          kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(64,
          kernel_initializer = 'truncated_normal',
          bias_initializer='zeros',
          use_bias = True,
          activation = 'linear',
          kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)

#x = Dropout(0.5)(x)
angle_out = Dense(units=1, 
                  kernel_initializer = 'truncated_normal',
                  bias_initializer='zeros',
                  use_bias = True,
                  activation='linear', 
                  name='angle_out')(x)
#x = BatchNormalization()(x)
#speed_out = Dense(units=1, activation='linear', name='speed_out')(x)

#model = Model(inputs=[input_image], outputs=[angle_out, speed_out])
model = Model(inputs=[input_image], outputs=[angle_out])

folders = ['AlexT-1', 'AlexT-2', 'AlexT-3', 'AlexT-4']
df = gd.assemble_data(folders)
df = gd.clean_dataframe(df)
df = df.sample(frac=1).reset_index(drop=True)


initial_batch_size = 128
epochs = 5

#batch_size = 128
train_cost = []
run_number = 0
batch_size = 128
validation_batch_size = 100
steps_per_epoch = 256
train_iter = gd.gen_batch(df,batch_size=batch_size)

validation_iter = gd.gen_batch(df,batch_size=validation_batch_size)  
X_valid_images, y_valid_angles, y_valid_speeds = next(validation_iter)      
#epoch = 2
#history = model.fit_generator(train_iter, steps_per_epoch=steps_per_epoch, 
#                              epochs=epochs,
#                              validation_data=(X_valid_images, y_valid_angles), 
#                              verbose=1)



#fit_generator(<generator..., validation_data=(array([[[..., verbose=1, steps_per_epoch=248, epochs=2)`
for epoch in range(epochs):
   batch_size = initial_batch_size + epoch*64
   batch_size = np.min([batch_size, 256])
   print("Training model.  Epoch# ", epoch, " Batch size: ", batch_size)  
   for batch in range(4000//batch_size):
     run_number += 1
     train_iter = gd.gen_batch(df,batch_size=batch_size)
     X_train, y_train_angle, y_train_speed = next(train_iter)
     #Model.train_on_batch(model, X_train, [y_train_angle, y_train_speed])
     train = model.train_on_batch(X_train, y_train_angle)
     train_cost.append(train)
        
     if run_number%20 == 0:
        print("cost (last 10): ", np.mean(train_cost[-10:]),   
              " batch: ", batch+1)
 
        #steering_angle = float(model.predict(X_valid, batch_size=128))
        X_valid_images, y_valid_angles, y_valid_speeds = next(validation_iter)            
        predicted_angles = model.predict(X_valid_images)
        #errors = np.square(y_valid_angles - steering_angles)
        
        for truth_angle, predicted_angle in zip(y_valid_angles, predicted_angles):
            print("Angle prediction:" , predicted_angle, 
                  " Ground Truth: ", truth_angle,
                  " Error: ", truth_angle-predicted_angle)
            
           
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

