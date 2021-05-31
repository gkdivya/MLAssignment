
# ML Assignment - Person Car detection

Objective is to train a model to localize and classify each instance of person and car in images using Object detection algorithm.

## Dataset
Path: [Person Car Dataset](https://evp-ml-data.s3.us-east-2.amazonaws.com/mlinterview/openimages-personcar/trainval.tar.gz)

### Dataset Structure:

    trainval/
      images/
        image_000000001.jpg
        image_000000001.jpg
        image_000000001.jpg
        image_000000001.jpg
        ...
      annotations/
        bbox-annotations.json
      
### Analysis:
    -   No of Unique Images:
    -   No of Annotations:
    -   Categories:
    -   Improper Annotation: Image id: 1378 was wrongly annotated

### Custom Dataset:
Defined custom dataset which takes the dataset path and the annotation file as input. All the training images are inside the “train-val” folder. 
</br>
In the __getitem__ method, read the image using the image_id and all the meta data associated with that image.
Initialized a dictionary called 'data_annotation', which will be passed to the model for training. 
</br>
    This dictionary will have all the metadata of the annotation like 
    <ul>
        <li>actual bounding box coordinates</li>
        <li>it’s corresponding labels</li>
        <li>image_id</li>
        <li>area of the bounding boxes. (Area param is used during evaluation with the COCO metric, to separate the metric scores between small, medium, and large boxes)</li>
        <li>iscrowd   (Instances with isCrowd as 'True' will be ignored during evaluation</li>
    </ul>
In the __len__ method, the size of the Dataset is returned.

### Dataloader: 
Data loader will load the training data in batches into the model for training. Using PyTorch’s DataLoader utility, dataset was split into train and val sets.
</br>

### Model: 
Torchvision’s FasterRCNN with a resnet50 backbone is used with pretrained weights as false to train and predict persons and cars in the images.
</br>
</br>
[Person Car Detection Colab Notebook](https://colab.research.google.com/github/gkdivya/MLAssignment/blob/main/PersonCar_Detection.ipynb)

## Experiments

Batch Size = 16 <br>
Epochs = 5

|Experiment| Batch Size | Epochs | Augmentation | Learning Rate Scheduler | Final loss| Status | 
|-------|---|---|---|---|---|---|
|[Base Skeleton Model](https://github.com/gkdivya/MLAssignment/blob/main/Experiments/Base_Skeleton_Model_PersonCar_Detection.ipynb) |16|5|No|SGD 0.01| 0.87 | Completed | 
|[With Step LR change]() |16|5|No|SGD 0.01||In progress  | 
|[With Image Augmentation]() |16|5|Yes|SGD 0.01||In progress  | 

### Model Architecture

![image](https://user-images.githubusercontent.com/17870236/120178750-71947580-c227-11eb-9432-77e7f455a945.png)

### Training Log

    Epoch: 0, Loss: 1.228668451309204
    Epoch: 1, Loss: 1.333381175994873
    Epoch: 2, Loss: 1.1773134469985962
    Epoch: 3, Loss: 0.730384349822998
    Epoch: 4, Loss: 0.33463841676712036 

### Validation Output
![image](https://user-images.githubusercontent.com/17870236/120183882-eb2f6200-c22d-11eb-98ed-04693f38ea12.png)


## Reference Links
[Kaggle Notebook]( https://www.kaggle.com/bharatb964/pytorch-implementation-of-faster-r-cnn) </br>
[PyTorch TorchVision Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)</br>
[Faster Rcnn](https://blog.francium.tech/object-detection-with-faster-rcnn-bc2e4295bf49)</br>
