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
