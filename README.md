# Freespace Segmentation
The project aims to determine the drivable area with semantic segmentation.

##  Main topics of the project
The project consists of 6 main titles. These:
 - json2mask
 - preprocess
 - model
 - train
 - augmention

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
