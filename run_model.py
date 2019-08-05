import numpy as np
from keras.models import load_model
import h5py
import generate_data as gd


model = load_model('./model.h5')
folders = ['AlexT-1', 'AlexT-2', 'AlexT-3']

df = gd.assemble_data(folders)
df = gd.clean_dataframe(df)
df = df.sample(frac=1).reset_index(drop=True)


validation_batch_size = 100
validation_iter = gd.gen_batch(df,batch_size=validation_batch_size)  

prediciton_angle_means = []
 
#steering_angle = float(model.predict(X_valid, batch_size=128))
X_valid, y_valid_angle, y_valid_speed = next(validation_iter)            
steering_angle = Model.predict(model, X_valid)
prediction_error = y_valid_angle - steering_angle
        
for i in range(100):
    if y_valid_angle[i] > .1:
       print("Right BIG ONE:  Angle prediction:" , steering_angle[i], 
             " Ground Truth: ", y_valid_angle[i],
             " Prediction Error: ", y_valid_angle[i] - steering_angle[i])
   
 
