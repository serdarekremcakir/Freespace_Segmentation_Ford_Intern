#Gerekli kütüphaneler import edildi
from modelnew import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import torch
import torch.nn as nn
import torch.optim as optim
from parameters import *
from matplotlib import pyplot as plt


#Görüntü ve maske görüntülerinin aynı isimde olup olmadığı kontrol ediliyor
image_mask_check(image_path_list, mask_path_list)


#Oluşturduğumuz modeli gerekli parametreleri vererek çağırıyoruz
model = FoInternNet(n_channels=3, n_classes=2, bilinear=True)

#Kayıp fonksiyonu ve optimizasyon fonksiyonları belirlenir
criterion = nn.BCELoss() #Binary Cross Entropy Loss kısaltmasıdır.
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()


# TRAINING THE NEURAL NETWORK
## grafik çizmek için (değerleri kaydetmek için) gerekli listeler
val_loss_save=[]
run_loss_save=[]

#Gerekli eğitim işlemleri yapılıyor ve validation loss değerleri ve train loss değerleri kaydediliyor
for epoch in range(epochs):
    running_loss = 0
    for ind in range(steps_per_epoch):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)

    
        optimizer.zero_grad()
        outputs = model(batch_input)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        #print(ind)
        if ind == steps_per_epoch-1:
            run_loss_save.append(running_loss)
            print('training loss on epoch {}: {}'.format(epoch, running_loss))
            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                val_loss += loss
                break
            val_loss_save.append(val_loss)
            print('validation loss on epoch {}: {}'.format(epoch, val_loss))

#Eğitilen model kaydediliyor
torch.save(model, 'drive/MyDrive/InternP1/serdar/serdar-model.pt')


#Grafik
epoch_list=list(range(1, epochs+1))
norm_validation = [float(i)/sum(val_loss_save) for i in val_loss_save]
norm_train = [float(i)/sum(run_loss_save) for i in run_loss_save]
plt.figure(figsize=(16,8))
plt.subplot(221)
plt.plot(epoch_list, norm_train,color="red") 
plt.title("Train Loss", fontsize=13)

plt.subplot(222)
plt.plot(epoch_list, norm_validation, color="blue") 
plt.title("Validation Loss", fontsize=13)

plt.subplot(212)
plt.plot(epoch_list, norm_train,  color="red") 
plt.plot(epoch_list, norm_validation, color="blue") 
plt.title("Train and Validation Loss", fontsize=13)
plt.savefig("drive/MyDrive/InternP1/serdar/serdar-loss-normalization.png")
plt.show()
