import numpy as np
import cv2
import os
import tqdm 
from torchvision import transforms
from PIL import Image
import glob

valid_size = 0.3 #Modeli değerlendirmek için kullanılacak validation setin oranı
test_size  = 0.1 #Test edilecek verisetinin oranı

DATA_DIR = os.path.join('../data_test')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')

augimage = '../data_test/augimage/'
if not os.path.exists(augimage):
    os.mkdir(augimage)
    
augmask = '../data_test/augmask/'
if not os.path.exists(augmask):
    os.mkdir(augmask)
###############################

# PREPARE IMAGE AND MASK LISTS

#Görüntü ve maskelerin isimleri listeye alınıyor ve sıralanıyor
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()
mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

indices = np.random.permutation(len(image_path_list))
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]


deneme = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06),
        #transforms.ToTensor()
    ])


def imageaug():
    uzunluk = len (train_input_path_list)
    i = 0
    for image in tqdm.tqdm(train_input_path_list):
        q1 = image[20:-4] + "7.jpg"
        q2 = augimage + q1
        x = Image.open(image)
        x = deneme(x)
        x = np.array(x)
        cv2.imwrite(q2,x)
        
        i +=1
        train_input_path_list.append(q2)
        if uzunluk <= i:
            break;
    return train_input_path_list
        

def maskaug():
    uzunluk = len (train_label_path_list)
    i = 0
    for mask in tqdm.tqdm(train_label_path_list):
        maske=cv2.imread(mask)
        q3 = mask[19:-4] + "7.png"
        q4 = augmask + q3
        cv2.imwrite(q4,maske)
        
        i +=1
        train_label_path_list.append(q4)
        if uzunluk <= i:
            break;
    
    return train_label_path_list

