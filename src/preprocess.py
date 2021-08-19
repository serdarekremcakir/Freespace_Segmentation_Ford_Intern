"""
We need to prepare the data to feed the network: we have - data/masks, data/images - directories where we prepared masks and input images. Then, convert each file/image into a tensor for our purpose.

We need to write two functions in src/preprocess.py:
    - one for feature/input          -> tensorize_image()
    - the other for mask/label    -> tensorize_mask()


Our model will accepts the input/feature tensor whose dimension is
[batch_size, output_shape[0], output_shape[1], 3]
&
the label tensor whose dimension is
[batch_size, output_shape[0], output_shape[1], 2].

At the end of the task, our data will be ready to train the model designed.

"""

#Kütüphaneler import edildi
import glob
import cv2
import torch
import numpy as np
from constant import *


def tensorize_image(image_path_list, output_shape, cuda=False):
    """
    Parameters
    ----------
    image_path_list : list of strings
        [“data/images/img1.png”, .., “data/images/imgn.png”] corresponds to
        n images to be trained each step.
    output_shape : tuple of integers
        (n1, n2): n1, n2 is width and height of the DNN model’s input.
    cuda : boolean, optional
        For multiprocessing,switch to True. The default is False.

    Returns
    -------
    torch_image : Torch tensor
        Batch tensor whose size is [batch_size, output_shape[0], output_shape[1], C].       For this case C = 3.
    """
    #Boş bir liste oluşturuldu
    local_image_list = []

    #Her resim için
    for image_path in image_path_list:

        #Görüntü okundu
        image = cv2.imread(image_path)

        #Görüntü yeniden boyutlandırıldı
        image = cv2.resize(image, output_shape)
        
        """
        print("image_path \n",image)
        print("deneme \n",image[:,:,0])
        print("image[:,:,0].shape",image[:,:,0].shape)
        print(image.shape)
        print(image.dtype)
        """
        
        #Input yapısı pytorch Input yapına göre değiştirildi
        torchlike_image = torchlike_data(image)

        #Listeye eklendi
        local_image_list.append(torchlike_image)

    #Liste yapısından torch tensoruna dönüştürüldü.
    image_array = np.array(local_image_list, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()
    

    # If multiprocessing is chosen
    if cuda:
        torch_image = torch_image.cuda()
        
    return torch_image
    
def tensorize_mask(mask_path_list, output_shape, n_class, cuda=False):

    #Boş liste oluşturuldu
    local_mask_list = []

    #Her maske için
    for mask_path in mask_path_list:

        #Maske okundu
        mask = cv2.imread(mask_path, 0)

        #Yeniden boyutlandırıldı
        mask = cv2.resize(mask, output_shape)

        #One-hot encoding uygulandı
        mask = one_hot_encoder(mask, n_class)

        #Input yapısı pytorch input yapına göre değiştirildi
        torchlike_mask = torchlike_data(mask)

        #Listeye eklendi
        local_mask_list.append(torchlike_mask)

    #Liste yapısından torch tensoruna dönüştürüldü.
    mask_array = np.array(local_mask_list, dtype=np.int)
    torch_mask = torch.from_numpy(mask_array).float()
    
    if cuda:
        torch_mask = torch_mask.cuda()

    return torch_mask

def image_mask_check(image_path_list, mask_path_list):

    # Dosya yolu listelerinin uzunluğu kontrol edilir
    if len(image_path_list) != len(mask_path_list):
        print("There are missing files ! Images and masks folder should have same number of files.")
        return False

    # İsimlerin aynı olup olmadığı kontrol edilir
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image_name = image_path.split('\\')[-1].split('.')[0]
        mask_name  = mask_path.split('\\')[-1].split('.')[0]
        if image_name != mask_name:
            print("Image and mask name does not match {} - {}".format(image_name, mask_name)+"\nImages and masks folder should have same file names." )
            return False

    return True

############################ TODO ################################
def torchlike_data(data):
    """
    Change data structure according to Torch Tensor structure where the first
    dimension corresponds to the data depth.


    Parameters
    ----------
    data : Array of uint8
        Shape : HxWxC.

    Returns
    -------
    torchlike_data_output : Array of float64
        Shape : CxHxW.

    """

    #Inputun kanal değeri alınır
    n_channels = data.shape[2]    ### 3
   # print("n_channels=", n_channels)  #(224,224,3)

    #Inputla benzer türde ve boyutta boş bir dizi oluşturuluyor.
    torchlike_data_output = np.empty((n_channels,data.shape[0],data.shape[1]))
   # print("torchlike_data_output",torchlike_data_output[0])
   
    #print("xxxxxxxxxxxxxxx",torchlike_data_output[0].shape)    
    #
    #
    #numpy.empty(shape, dtype=float, order='C', *, like=None)
    
    
    #her kanal için
    for i in range(n_channels):
        
        #datanın niçindeki veriler torchlika_data_output a aktarılır
        torchlike_data_output[i] = data[:,:,i]
    return torchlike_data_output

def one_hot_encoder(data, n_class):
    """
    Returns a matrix containing as many channels as the number of unique
    values ​​in the input Matrix, where each channel represents a unique class.


    Parameters
    ----------
    data : Array of uint8
        2D matrix.
    n_class : integer
        Number of class.

    Returns
    -------
    encoded_data : Array of int64
        Each channel labels for a class.

    """
    if len(data.shape) != 2:
        print("It should be same with the layer dimension, in this case it is 2")
        return
    if len(np.unique(data)) != n_class:
        print("The number of unique values ​​in 'data' must be equal to the n_class")
        return

    # Boyutu (genişlik, yükseklik, sınıf_sayısı) olan diziyi tanımlandı
    encoded_data = np.zeros((*data.shape, n_class), dtype=np.int)

    # Etiker(label) tanımlandı
    encoded_labels = [[0,1],[1,0]]

    #
    for lbl in range(n_class):
        encoded_label = encoded_labels[lbl]
        # lbl = 0 için (arkaplan) [1, 0] labelini oluşturuyorum, 
        # lbl = 1 için (freespace) [0, 1] labelini oluşturuyorum.
        
        numerical_class_inds = data[:,:] == lbl 
        # lbl = 0 için data'nın 0'a eşit olduğu w,h ikililerini alıyorum diyelim ki (F).
        # lbl = 1 için data'nın 1'e eşit olduğu w,h ikililerini alıyorum diyelim ki (O).
        
        encoded_data[numerical_class_inds] = encoded_label 
        # lbl = 0 için tüm F'in sahip olduğu tüm w,h ikililerini [1, 0]'a eşitliyorum.
        # lbl = 1 için tüm O'un sahip olduğu tüm w,h ikililerini [0, 1]'e eşitliyorum.
        
    return encoded_data
############################ TODO END ################################



if __name__ == '__main__':

    # Görüntüler okundu ve sıralandı
    image_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
    image_list.sort()

    # Maskeler okundu ve sıralandı
    mask_list = glob.glob(os.path.join(MASK_DIR, '*'))
    mask_list.sort()


    # Görüntü ve maskelerin isimlerinin kontrolü yapıldı
    if image_mask_check(image_list, mask_list):

        # Belirtilen batch_size değişkeni kadar görüntü bir listeye kaydedildi
        batch_image_list = image_list[:BACTH_SIZE]

        # Torch tensorune dönüştürüldü
        batch_image_tensor = tensorize_image(batch_image_list, (224, 224))

        # Kontroller
        print("For features:\ndtype is "+str(batch_image_tensor.dtype))
        print("Type is "+str(type(batch_image_tensor)))
        print("The size should be ["+str(BACTH_SIZE)+", 3, "+str(HEIGHT)+", "+str(WIDTH)+"]")
        print("Size is "+str(batch_image_tensor.shape)+"\n")

        # Belirtilen batch_size değişkeni kadar maske bir listeye kaydedildi
        batch_mask_list = mask_list[:BACTH_SIZE]

        # Torch tensorune dönüştürüldü
        batch_mask_tensor = tensorize_mask(batch_mask_list, (HEIGHT, WIDTH), 2)

        # Kontrol
        print("For labels:\ndtype is "+str(batch_mask_tensor.dtype))
        print("Type is "+str(type(batch_mask_tensor)))
        print("The size should be ["+str(BACTH_SIZE)+", 2, "+str(HEIGHT)+", "+str(WIDTH)+"]")
        print("Size is "+str(batch_mask_tensor.shape))

