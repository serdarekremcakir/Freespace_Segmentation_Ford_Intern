#Kütüphaneler import edildi
import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constant import *


#Maskeler klasöründeki her dosyanın adını içeren liste oluşturuldu
mask_list = os.listdir(MASK_DIR)

#Gizli dosyalar varsa kaldırıldı
for f in mask_list:
    if f.startswith('.'):
        mask_list.remove(f)

#Her maske görüntüsü için
for mask_name in tqdm.tqdm(mask_list):
    
    #Gizli uzantılar yok sayıldı
    mask_name_without_ex = mask_name.split('.')[0]

    #Gerekli dosya yolları tanımlandı
    mask_path      = os.path.join(MASK_DIR, mask_name)
    image_path     = os.path.join(IMAGE_DIR, mask_name_without_ex+'.jpg')
    image_out_path = os.path.join(IMAGE_OUT_DIR, mask_name)

    #Maske ve ilgili görüntü çağrıldı
    mask  = cv2.imread(mask_path, 0).astype(np.uint8)
    image = cv2.imread(image_path).astype(np.uint8)

    #Görüntünün kopyası oluşturuldu.
    #Bu imagede mask değerlerinin 1 olduğu yerlerin rengi değiştirildi.
    #Orjinal görüntü ile maskelenmiş görüntü birleştiriliyor.
    cpy_image  = image.copy()
    image[mask==1, :] = (255, 0, 125)
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)

    #Maske uygulanmış görüntü kaydedilir.
    cv2.imwrite(image_out_path, opac_image)

    #constant.py de visualize true ise maskeli görüntüler gösterilecek
    if VISUALIZE:
        plt.figure()
        plt.imshow(opac_image)
        plt.show()


