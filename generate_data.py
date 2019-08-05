import pandas
from pandas import DataFrame
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import itertools
from scipy.stats import bernoulli
from skimage.util import random_noise
from scipy.signal import gaussian
from skimage.util import random_noise
from PIL import Image, ImageEnhance
    
# assemble the various data collecton runs
#note: this is a little better than combining one time because
#new datasets may be forthcoming
#function called on the first run of gen_batch to provide a list of samples

def assemble_data(folders):
  #kernal used to smooth driver angle data 
  kernel = gaussian(5, 1)  
  all_images, all_angles, all_throttles, all_brakes, all_speeds, = [],[],[],[],[]
  #loop over the folders containing the acquired data
  for folder in folders:
     #print('processing folder: ', folder)
     csv_data = pandas.read_csv(folder + '/driving_log.csv')
     center = csv_data.iloc[:,0]
     left = csv_data.iloc[:,1]
     right = csv_data.iloc[:,2]
     image_names = list(itertools.chain(center, left, right))
     for image_name in image_names:       
         #rename the images to the local filesystem
         corrected = image_name.replace('/Users/alex/Desktop/data', './'+folder)
         corrected = corrected.replace('/root/Desktop', './'+folder)
         #print(corrected)
         all_images.append(corrected)
 
     #a little smoothing with remove jerkiness in acquired data
     angles = csv_data.iloc[:, 3]
     #print(np.mean(angles))
     smooth = np.convolve(angles, kernel, "same") / sum(kernel)
     #smooth = angles
     ##make three copies of smooth angles, throttles, brakes, speeds
     ##to match center, left and right images
     all_angles.append(list(itertools.chain(angles, angles, angles)))
     throttles = csv_data.iloc[:, 4]            
     all_throttles.append(list(itertools.chain(throttles,throttles,throttles)))
     brakes = csv_data.iloc[:, 5]
     all_brakes.append(list(itertools.chain(brakes,brakes,brakes)))
     speeds  = csv_data.iloc[:, 6] 
     all_speeds.append(list(itertools.chain(speeds,speeds,speeds)))
        
  #3user itertools chain.from_iterable to flatten the individual datasets     
  all_angles = list(itertools.chain.from_iterable(all_angles))
  all_throttles = list(itertools.chain.from_iterable(all_throttles))
  all_brakes = list(itertools.chain.from_iterable(all_brakes))
  all_speeds = list(itertools.chain.from_iterable(all_speeds))

  ##created a dataframe with a column for splits
  Data = {'images':  all_images, 'angles': all_angles,
          'throttles': all_throttles, 'brakes': all_brakes,
          'speeds': all_speeds}
 
  df = DataFrame (Data, columns = ['images','angles','throttles',
                                   'brakes', 'speeds'])
  
  return df


def flip_coin(prob=0.5):
    return True if bernoulli.rvs(prob) == 1 else False  
def clean_dataframe(df):
   #print(np.max(df['angles']))
   #print(df)

   #eliminate big turns on the entire dataset
   # adjust this parameter if the car goes too straight
   bigturn_index = df[df['angles'].abs() > 1.0].index
   df.drop(bigturn_index, inplace=True)
   #print(df)
   #find appropriate images from left and right cameras to use.  Consider the following:
   #a left camera view when the car is pointed way left is not useful since the only camera
   #we train for is a central camera.  Likewise for far right images.
   #left_camera_index = df[df['images'].str.contains('left')].index
   #right_camera_index = df[df['images'].str.contains('right')].index
   #bigturn_index = df[df['angles'].abs() > .4].index
   #df.drop(bigturn_index & left_camera_index, inplace=True)
   #df.drop(bigturn_index & right_camera_index, inplace=True)

   left_camera_index = df[df['images'].str.contains('left')].index
   right_camera_index = df[df['images'].str.contains('right')].index

   #df.loc[left_camera_index, 'angles'] *= -1.0
   df.loc[left_camera_index, 'angles'] = 0.25

   #df.loc[right_camera_index, 'angles'] *= -1.0
   df.loc[right_camera_index, 'angles'] = -0.25


   ##add a rank column for elimination of the angesl that are too large
   #df['rank']=df.angles.rank(method='first').astype(int)
   #df['inv_rank']=df.angles.rank(ascending=False, method='first').astype(int)
    
   return df

def gen_image(df, candidates, generate=True):
    #returns a single instance 
    #print(selection)
    sample = candidates.sample()
    
    angle = sample['angles'].tolist()[0]
    speed = sample['speeds'].tolist()[0]
    imname = sample['images'].tolist()[0]
    

    #base image before augmentation
   # print(imname)
    img = Image.open(imname)
    
    if generate: 
      all_angles = df['angles']
    
      if flip_coin():
         brightness_enhancer = ImageEnhance.Brightness(img)
         img = brightness_enhancer.enhance(random.uniform(.4,1.4))
        
      if flip_coin():
          search_angle = angle * -1
          #print("search angle", search_angle)
          index = (np.abs(all_angles-search_angle)).idxmin()
          flipper = df[df['angles'] == all_angles[index]]
          imname = flipper['images'].tolist()[0]
   
          img = Image.open(imname)  
          img = img.transpose(Image.FLIP_LEFT_RIGHT)
          angle = search_angle
      
      ##apply random gamma brightness adjust
        
 
      if flip_coin(): 
         noise = random_noise(np.asarray(img), mode='s&p')
         img = (np.array(255*noise, dtype = 'uint8')) 
        
      if flip_coin(): 
         noise = random_noise(np.asarray(img), mode='poisson')
         img = (np.array(255*noise, dtype = 'uint8')) 
        
      if flip_coin(): 
         noise = random_noise(np.asarray(img), mode='gaussian')
         img = (np.array(255*noise, dtype = 'uint8')) 
  
    return img, angle, speed
 
 
def gen_batch(df, batch_size, split='train'):

  #angles = df['angles'].tolist()

  selection_ranges = np.arange(-.35,.351,.05)
  select_high = selection_ranges[1:]
  select_low = selection_ranges[:-1]
 # print(select_high, select_low)
  
  hist,centers = np.histogram(df['angles'],len(select_high))

  #print(np.max(hist))
  class_percentages = hist/np.max(hist)
 # print("CLASS:" ,class_percentages)
#  print(class_percentages)
  while True:
    class_select = [random.randint(0,len(select_high)-1) for _ in range(batch_size)]
 
    features = np.zeros((batch_size,160,320,3),dtype='uint8')
    angles = np.zeros(batch_size)
    speeds = np.zeros(batch_size)

    for indx,select in enumerate(class_select):
        
       candidates = df[(df['angles'] >= select_low[select]) & (df['angles'] < select_high[select])] 
     #  print("*********88888888**********")
     #  print("select", select, "select low", select_low[select])
     #  print(candidates)
       ##the following genereates augmented images with p==1-class percent
       generate_fake = flip_coin(class_percentages[select])      
       img, angle, speed = gen_image(df, candidates, generate=generate_fake)

       #print("", angle)   
       features[indx,:,:,:] = img
       angles[indx] = angle
       speeds[indx] = speed

    yield (features, angles, speeds)
               
        

