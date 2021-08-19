#Kütüphaneler import edildi
import json
import os
import numpy as np
import cv2
import tqdm
from constant import JSON_DIR, MASK_DIR


# "jsons" klasöründeki her dosya adını içeren bir liste oluşturuldu
json_list = os.listdir(JSON_DIR)

""" tqdm Example Start"""
iterator_example = range(1000000)

for i in tqdm.tqdm(iterator_example):
    pass
""" rqdm Example End"""


#Her json dosyası için 
for json_name in tqdm.tqdm(json_list):

    #json dosyalarının bulunduğu klasör ile her json dosyasının isimini birleştirerek bir yol oluşturuldu
    json_path = os.path.join(JSON_DIR, json_name)
    
    #"r" ile sadece okumak istediğimizi belirterek açtık
    json_file = open(json_path, 'r')
    

    #json dosyasının içindeki tüm bilgiler sözlük yapısında açıldı
    json_dict = json.load(json_file)

    #0 lardan oluşan bir matris oluşturuldu. Matrisin boyutları etiketlenmiş resimlerden elde ediliyor.
    #Sözlük yapısında resim boyutları gözlemlenebilir.
    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    
    #Oluşturulan maskenin kaydedilmesi için bir dosya yolu oluşturuldu.
    mask_path = os.path.join(MASK_DIR, json_name[:-9]+".png")

    #Json içindeki her obje için
    for obj in json_dict["objects"]:
    
        #Objenin classTitle'inin Freespace olup olmadığı kontrol edildi.
        if obj['classTitle']=='Freespace':
        
            #Pointleri teker teker çekilerek oluşturulan boş matrisin içerisine çizdirilir.
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)

    #Maske kaydedilir.
    cv2.imwrite(mask_path, mask.astype(np.uint8))