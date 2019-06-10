import pandas as pd 
import numpy as np
import os
import h5py
from tqdm import tqdm


def create_and_store_X_train():
    storage_path = os.path.join(os.getcwd(),'data','X_train.h5')

    #Check if file is already stored into path. If not, create and store them
    if not os.path.isfile(storage_path):
        train_path = os.path.join(os.getcwd(),'data','training.csv') #training.csv path
        train_csv = pd.read_csv(train_path, engine='python')

        image_pixels_str = train_csv.Image #Select dataframe column that stores image pixels in a str

        #For each image pixels str, split it into a list of str pixels. For example --> ['1 2 3']-->['1','2','3']
        images_list = []
        for image in tqdm(image_pixels_str):
            image = image.split(' ')
            for pixel in tqdm(image):
                if pixel =='': #For simplicity, if we find and empty pixel value, fill this with '0'. Proposal: interpolate it with pixel values of the neighborhood
                    pixel = '0'
            images_list.append(image) #Append into a list. The result is a list images and inside each row, a list o pixels
        
        images_matrix = np.array(images_list, dtype = 'float') #Convert into a numpy array
        X_train = images_matrix.reshape(-1,96,96) #Reshape to image resolution of (96,96).

        #Store it in order to avoid repeating calculations
        hf = h5py.File(storage_path,'w')
        hf.create_dataset(
            name='X_train',
            data=X_train,
            dtype=np.float32 
        )
        hf.close()

def create_and_store_X_test():
    storage_path = os.path.join(os.getcwd(),'data','X_test.h5')

    #Check if file is already stored into path. If not, create and store them
    if not os.path.isfile(storage_path):
        test_path = os.path.join(os.getcwd(),'data','test.csv') #test.csv path
        test_csv = pd.read_csv(test_path, engine='python')

        image_pixels_str = test_csv.Image #Select dataframe column that stores image pixels in a str

        #For each image pixels str, split it into a list of str pixels. For example --> ['1 2 3']-->['1','2','3']
        images_list = []
        for image in tqdm(image_pixels_str):
            image = image.split(' ')
            for pixel in tqdm(image):
                if pixel =='': #For simplicity, if we find and empty pixel value, fill this with '0'. Proposal: interpolate it with pixel values of the neighborhood
                    pixel = '0'
            images_list.append(image) #Append into a list. The result is a list images and inside each row, a list o pixels
        
        images_matrix = np.array(images_list, dtype = 'float') #Convert into a numpy array
        X_test = images_matrix.reshape(-1,96,96) #Reshape to image resolution of (96,96).

        #Store it in order to avoid repeating calculations
        hf = h5py.File(storage_path,'w')
        hf.create_dataset(
            name='X_test',
            data=X_test,
            dtype=np.float32 
        )
        hf.close()
    
def create_and_store_y_train():
    storage_path = os.path.join(os.getcwd(),'data','y_train.h5')

    #Check if file is already stored into path. If not, create and store them
    if not os.path.isfile(storage_path):
        train_path = os.path.join(os.getcwd(),'data','training.csv') #training.csv path
        train_csv = pd.read_csv(train_path, engine='python')

        train_csv = train_csv.drop(labels='Image', axis=1) #Drop image pixels column in order to keep only keypoints data

        #Check if there are null values
        #We use forward filling but there are more options. Proposal: Study pandas fillna method options and check which option is more suitable with our problem
        if train_csv.isnull().any().sum() != 0:
            train_csv.fillna(method = 'ffill',inplace = True)

        y_train = np.empty([train_csv.shape[0],train_csv.shape[1]]) #Create an empty matrix of y_train shape
        for i in tqdm(range(train_csv.shape[0])):
            y_train[i] = train_csv.iloc[i,:].values #Fill the empty matrix with y_train of each row in dataframe

        #Store it in order to avoid repeating calculations
        hf = h5py.File(storage_path,'w')
        hf.create_dataset(
            name='y_train',
            data=y_train,
            dtype=np.float32 
        )
        hf.close()

def load_data():
    #Load X_train
    X_train_path = os.path.join(os.getcwd(),'data','X_train.h5')
    hf = h5py.File(X_train_path,'r')
    X_train = hf['X_train'][:]
    hf.close()

    #Load y_train
    y_train_path = os.path.join(os.getcwd(),'data','y_train.h5')
    hf = h5py.File(y_train_path,'r')
    y_train = hf['y_train'][:]
    hf.close()

    #Load X_test
    X_test_path = os.path.join(os.getcwd(),'data','X_test.h5')
    hf = h5py.File(X_test_path,'r')
    X_test = hf['X_test'][:]
    hf.close()

    return X_train, y_train, X_test
    
if __name__ == "__main__":
    create_and_store_X_train()
    create_and_store_X_test()
    create_and_store_y_train()
    X_train, y_train, X_test = load_data()
    #Here we should create model design and training part
    #Last step consists on predict X_test images and submit on Kaggle