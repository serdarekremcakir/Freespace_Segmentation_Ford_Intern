import os
import glob
import numpy as np

######### PARAMETERS ##########
valid_size = 0.3 #Modeli değerlendirmek için kullanılacak validation setin oranı
test_size  = 0.1 #Test edilecek verisetinin oranı
batch_size = 8 #Modelin kaç fotoğrafı aynı anda işleyeceğinin sayısı
epochs = 2 #Kaç eğitim adımı olacağının sayısı
cuda = False
input_shape = (224, 224) #Görüntülerin modele giriş boyutu
n_classes = 2 #Sınıf sayısı
augmentiondene = False
###############################

######### DIRECTORIES #########

#Eğitim için gerekli dosya yolları
DATA_DIR = os.path.join('../data_test')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
PREDICT_DIR = os.path.join(DATA_DIR, 'predicts')
model_name = "serdar-model.pt"
model_path = os.path.join("../model", model_name)
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

#Test için ayrılacak olan görüntü ve maskeler belirleniyor
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

#Validation için ayrılacak olan görüntü ve maskeler belirleniyor
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

#Train için ayrılacak olan görüntü ve maskeler belirleniyor
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

if augmentiondene == True:
    from augmention import *
    train_input_path_list = imageaug()
    train_label_path_list = maskaug()
    
#Eğitim için ayrılmış veri sayısını yukarıda verilen batch_size sayısına bölerek
#Bir eğitimde(epoch) kaç yineleme yapıldığı bulunur
steps_per_epoch = len(train_input_path_list)//batch_size