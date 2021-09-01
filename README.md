# Freespace Segmentation
The project aims to determine the drivable area with semantic segmentation.

## Json to Mask
JSON files provide location information of points surrounding the area marked as drivable.


<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/json.png" title="hover text">
</p>


An empty mask was created to identify the freespace region and other regions in the images. This mask is filled with points in json files. This way json files are converted to mask images.

Mask Example: 


<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/foto1.jpg" width="400">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/mask1.png" width="400">

  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/foto2.jpg" width="400">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/mask2.png" width="400">
</p>

Codes of the part: [json2mask.py](https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/src/json2mask.py)

### Mask on Image
To test the accuracy of the masks created in the Json to Mask section, the masks were tested on the original images.

<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/maskesiz1.jpg" width="400">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/maskeli1.jpeg" width="400">

  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/maskesiz2.jpg" width="400">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/maskeli2.jpeg" width="400">

</p>

Codes of the part: [mask_on_image.py](https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/src/mask_on_image.py)

## Preprocess
Images and masks are used as input data to train the model.  Images and masks must be converted to Pytorch Tensor format.

Images are converted to a torch tensor:

	torchlike_image = torchlike_data(image)
    local_image_list.append(torchlike_image)
    
    image_array = np.array(local_image_list, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()


    def torchlike_data(data):
	    n_channels = data.shape[2]   
	    torchlike_data_output = np.empty((n_channels,data.shape[0],data.shape[1]))
	    
	    for i in range(n_channels):
	        torchlike_data_output[i] = data[:,:,i]
	    return torchlike_data_output



Size information of the generated tensor:

<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/tensor1.png" width="400">
</p>

The same processes are applied for masks, but there is a difference. The difference is one hot encoder. Mask are grayscale images. 

### ***What is grayscale image?***  
A grayscale image is simply one in which the only colors are shades of gray. The reason for differentiating such images from any other sort of color image is that less information needs to be provided for each pixel. 

### ***What is One Hot Encoding?***  
One Hot Encoding means that categorical variables are represented as binary. This operation requires mapping categorical values to integer values first. Then, each integer value is represented as a binary vector with all zero values except the integer index marked with 1

<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/onehotencoding.jpg" width="550">
</p>

 The operations applied to the pictures are also performed on the masks with the help of the function below.

    mask = one_hot_encoder(mask, n_class)
	
    torchlike_mask = torchlike_data(mask)
    local_mask_list.append(torchlike_mask)
	
    mask_array = np.array(local_mask_list, dtype=np.int)
    torch_mask = torch.from_numpy(mask_array).float()



    def one_hot_encoder(data, n_class):
	    encoded_data = np.zeros((*data.shape, n_class), dtype=np.int)
	    encoded_labels = [[0,1],[1,0]]

	    for lbl in range(n_class):
	        encoded_label = encoded_labels[lbl]
	        numerical_class_inds = data[:,:] == lbl 
	        encoded_data[numerical_class_inds] = encoded_label 
	    return encoded_data

Size information of the generated tensor:

<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/tensor2.png" width="400">
</p>

Codes of the part: [preprocess.py](https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/src/preprocess.py)

## Model
U-Net model was used in the project. The U-net model gives better results for semantic segmentation than other models, even with fewer photos.

The architecture contains two paths (Encoder and Decoder).

* The encoder is the first half in the architecture diagram. You apply convolution blocks followed by a maxpool downsampling to encode the input image into feature representations at multiple different levels.

* The decoder is the second half of the architecture. The goal is to semantically project the discriminative features (lower resolution) learned by the encoder onto the pixel space (higher resolution) to get a dense classification. The decoder consists of  **upsampling**  and  **concatenation**  followed by regular convolution operations.

<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/unet.png">
</p>

[Click for source](https://developers.arcgis.com/python/guide/how-unet-works/)


### ***What is Convolution?***  
Convolution is the simple application of a filter to an input that results in activation. Repeated application of the same filter to an input results in a map of activations called a feature map, indicating the locations and strength of a detected feature in input, such as an image.

<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/convolution.gif" width="500">
</p>

[Click for source](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)

### ***What is Max Pooling?***  
The function of pooling is to reduce the size of the feature map so that we have fewer parameters in the network. Basically from every kernel size block of the input feature map, we select the maximum pixel value and thus obtain a pooled feature map. 

<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/pooling.gif" width="500">
</p>

[Click for source](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)


### ***What is Activation Function?***  
In a neural network, the activation function is responsible for transforming the summed weighted input from the node into the activation of the node or output for that input.

### **ReLu activation function is used in this model.**

The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance.

<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/relu.png" width="665">
</p>

[Click for source](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)

I used the codes of a model that was working correctly, used before in a lot of projects.

[Click for the main source of the model used in the project](https://github.com/milesial/Pytorch-UNet)

Codes of the part: [modelnew.py](https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/src/modelnew.py)


## Train

Parameters required when training the model:

    ######## PARAMETERS ##########
    valid_size = 0.3 
    test_size = 0.1
    batch_size = 8
    epochs = 25
    cuda = False
    input_shape = (224, 224)
    n_classes = 25
    augmentiondene = False
    ###############################
        
File directories required when training the model:

    ######### DIRECTORIES #########
    DATA_DIR = os.path.join('../data_test')
    IMAGE_DIR = os.path.join(DATA_DIR, 'images')
    MASK_DIR = os.path.join(DATA_DIR, 'masks')
    PREDICT_DIR = os.path.join(DATA_DIR, 'predicts')
    model_name = "serdar-model.pt"
    model_path = os.path.join("../model", model_name)
    ###############################

Images and masks were prepared for model training. Train, validation and test data list were determined.

    image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
    image_path_list.sort()
    
    mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
    mask_path_list.sort()
    
    indices = np.random.permutation(len(image_path_list))
    
    test_ind = int(len(indices) * test_size)
    valid_ind = int(test_ind + len(indices) * valid_size)
    
    test_input_path_list = image_path_list[:test_ind]
    test_label_path_list = mask_path_list[:test_ind]
    
    valid_input_path_list = image_path_list[test_ind:valid_ind]
    valid_label_path_list = mask_path_list[test_ind:valid_ind]
    
    train_input_path_list = image_path_list[valid_ind:]
    train_label_path_list = mask_path_list[valid_ind:]

To be able to intervene more easily with the parameters and file paths required for model training, I have collected them in a single file.

Codes of the part:  [parameters.py](https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/src/parameters.py)

---

The model is called, the optimizer and the loss function are determined.

    model = FoInternNet(n_channels=3, n_classes=2, bilinear=True)
        
    criterion = nn.BCELoss() #Binary Cross Entropy Loss kısaltmasıdır.
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

The amount of data specified in the **"batch_size"** variable is taken from the training dataset and converted to the tensor format. The same process is done for masks. The selected data is given as input to the model and compared with the expected result. A loss value occurs as a result of the comparison. Depending on this loss value, **backpropagation** is done.  


    for  ind  in  range(steps_per_epoch):
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


Model is tested with validation data at the end of each epoch.

    for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
	    batch_input = tensorize_image([valid_input_path], input_shape, cuda)
	    batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
	    outputs = model(batch_input)
	    loss = criterion(outputs, batch_label)
	    val_loss += loss


These processes are repeated for the number as many **epochs**. Training loss and validation loss are saved in each epoch.

The saved loss values are visualized. The graph is drawn.


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
    plt.plot(epoch_list, norm_train, color="red")
    plt.plot(epoch_list, norm_validation, color="blue")
    plt.title("Train and Validation Loss", fontsize=13)
    plt.savefig("drive/MyDrive/InternP1/serdar/serdar-loss-normalization.png")
    plt.show()

The graph that emerged after the training:
<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/lossgraph.png" width="750">
</p>

Codes of the part:  [train.py](https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/src/train.py)


## Predict

The predictions of the trained model were observed test dataset.

The images in the test data set are converted to the tensor format. Data in tensor format is given to the model. Outputs from the model are converted to masks. Parts of mask images designated as drivable areas are converted to purple and added to the original test data.


    for image in range(images):
	    img = cv2.imread(image)
	    batch_test = tensorize_image([image], input_shape, cuda)
	    output = model(batch_test)
	    out = torch.argmax(output, axis=1)
	    out = out.cpu()
	    outputs_list = out.detach().numpy()
	    mask = np.squeeze(outputs_list, axis=0)
	    mask_uint8 = mask.astype('uint8')
	    mask_resize = cv2.resize(mask_uint8, ((img.shape[1]), (img.shape[0])), interpolation = cv2.INTER_CUBIC)
	    img_resize = cv2.resize(img, input_shape)
	    mask_ind = mask_resize == 1
	    copy_img = img.copy()
	    img[mask_resize==1, :] = (255, 0, 125)
	    opac_image = (img/2 + copy_img/2).astype(np.uint8)
	    cv2.imwrite(os.path.join(predict_path, image.split("/")[-1]), opac_image)




The model gave a good result on roads with bright, light traffic, and no objects such as cone and barrier.

Successfully predicted images:
<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/spre.png" width="600">
</p>

The model gave a bad result in conditions such as dark, tunnel, and under the bridge.

Unsuccessfully predicted images:
<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/unspre.png" width="750">
</p>


The model must be retrained with new data obtained after data augmentation to correct the unsuccessfully predicted images.


Codes of the part:  [predict.py](https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/src/predict.py)

## Data Augmention

The transform is defined to be used to change the brightness and contrast values of the data.

    deneme = transforms.Compose([
    transforms.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06),
    ])


		
The defined transform was applied to the data in the training dataset.  New images were saved in a different file with a new name. The file paths of the new images have been added to the train_input_path_list.

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

If the augmentation variable in the parameters.py file is selected as True, the model training will be done with the train dataset, which also contains the new data.

Augmented images:

<p align="center">
  <img src="https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/assets/aug.png" width="575">
</p>


Codes of the part:  [augmention.py](https://github.com/serdarekremcakir/Freespace_Segmentation_Ford_Intern/blob/main/src/augmention.py)