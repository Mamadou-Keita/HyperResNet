# HyperResNet

The HyperResNet model for color classification problem

![assets/HyperResNet.png](assets/HyperResNet.png)

## Dataset

> colorClassification

The datasets have contained about 80 images for trainset datasets for whole color classes and 90 images for the test set. colors which are prepared for this application is y yellow, black, white, green, red, orange, blue a and violet. In this implementation, basic colors are preferred for classification. and created a dataset containing images of these basic colors. ( [Link to dataset](https://www.kaggle.com/datasets/ayanzadeh93/color-classification) )

### Training

To train the model execute: 
```python
python train.py
```

### Testing

To test the model execute: 
```python
python test.py -m ./link to the model/model.pth -i ./test/red.png -n 9 
```
