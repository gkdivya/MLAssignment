
# ML Assignment - Person Car detection

Objective is to train a model to localize and classify each instance of person and car in images using Object detection algorithm.

## Dataset
[Dataset](https://evp-ml-data.s3.us-east-2.amazonaws.com/mlinterview/openimages-personcar/trainval.tar.gz)

*Dataset Structure:* 

    trainval/
      images/
        image_000000001.jpg
        image_000000001.jpg
        image_000000001.jpg
        image_000000001.jpg
        ...
      annotations/
        bbox-annotations.json
      
*Analysis:*
-   No of Unique Images:
-   No of Annotations:
-   Categories:

*Improper Annotation:*</br>
Image id: 1378 was wrongly annotated
</br>
*Custom Dataset:*
</br>
Defined custom dataset which takes the dataset path and the annotation file as input. All the training images are inside the “train-val” folder. 
</br>
In the __getitem__ method, read the image using the image_id and all the meta data associated with that image.
Initialized a dictionary called 'data_annotation', which will be passed to the model for training. 
This dictionary will have all the metadata of the annotation like 
    -   actual bounding box coordinates
    -   it’s corresponding labels
    -   image_id
    -   area of the bounding boxes. (Area param is used during evaluation with the COCO metric, to separate the metric scores between small, medium, and large boxes)
    -   iscrowd   (Instances with isCrowd as 'True' will be ignored during evaluation. 
In the __len__ method, the size of the Dataset is returned.
</br>
## Experiments

Batch Size = 16 <br>
Epochs = 5

|Experiment| Batch Size | Epochs | Augmentation | Learning Rate Scheduler | Final loss| Status | 
|-------|---|---|---|---|---|---|
|[Base Skeleton Model](https://github.com/gkdivya/MLAssignment/blob/main/Experiments/Base_Skeleton_Model_PersonCar_Detection.ipynb) |16|5|No|SGD 0.01| 0.87 | Completed | 
|[With Step LR change]() |16|5|No|SGD 0.01||In progress  | 
|[With Image Augmentation]() |16|5|Yes|SGD 0.01||In progress  | 

Below important concepts were used/considered while designing the network:
- PyTorch - Faster RCNN model is used for prediction
- Backbone model - resnet - 50
- 
[Person Car Detection Colab Notebook](https://colab.research.google.com/github/gkdivya/MLAssignment/blob/main/PersonCar_Detection.ipynb)
### Model Architecture

![image](https://user-images.githubusercontent.com/17870236/120178750-71947580-c227-11eb-9432-77e7f455a945.png)

### Training Log

    Epoch: 0, Loss: 1.7864028215408325
    Epoch: 1, Loss: 0.3092164397239685
    Epoch: 2, Loss: 0.3485269248485565
    Epoch: 3, Loss: 0.36063939332962036
    Epoch: 4, Loss: 0.8723194003105164

### Validation Output
![image](https://user-images.githubusercontent.com/17870236/120183882-eb2f6200-c22d-11eb-98ed-04693f38ea12.png)



## Reference Links
[Kaggle Notebook]( https://www.kaggle.com/bharatb964/pytorch-implementation-of-faster-r-cnn)
[PyTorch TorchVision Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
