# Image-Segmentation-UNet
Finetuned UNet model with ResNet backbone for segmentation on 24 test Pathology images, and reporting accuracy for each class.

First, I separated the Coloured and B&W images, as Images and Masks respectively. Since, Mask images are black & white, there are 2 classes. 
Next, we fine-tune the UNet model with ResNet backbone for 2 classes. 
60 images were taken as training dataset and since, the sample is small we took all 60 as validation dataset as well. 
Then, we trained the model for 20 epochs.
After, this model is evaluated for 24 images test dataset and the overall accuracy of the model is 85.29%, but the accuracy for Class 1 (Foreground) is low (63.09%), this is because we have only 60 images for training dataset. If we increase, the number of images in training dataset, both the overall model accuracy and Class1 accuracy will improve.
