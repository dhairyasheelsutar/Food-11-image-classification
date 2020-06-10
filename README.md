# Food 11 image classification

This is image classification on food 11 dataset. The dataset is taken from <a href='https://www.kaggle.com/vermaavi/food11'>https://www.kaggle.com/vermaavi/food11</a>. The dataset contains 16643 food images categorized into 11 foods. The 11 categories are Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. 

## Approach

<ul>
  <li>
    Train 3 pretrained models like vgg19, inceptionV3 and mobilenet on this dataset to find out which model works the best. 
  </li>
  <li>
    Evaluate all the models and select the model with highest accuracy on test set.  
   </li>
  <li>
    Use hyperparameter tuning to find out best parameters for the model. 
   </li>
  <li>
    Use error analysis to find out where the model is making mistakes and which class is contributing most of the error in the overall error. 
   </li>
  <li>
    Ensembling all the models to improve accuracy further.
  </li>
</ul>

## Experimental results

### Training models

Vgg19 model accuracy: 83.7%<br>
InceptionV3 accuracy: 85.3%<br>
Mobilenet accuracy: 83.2%<br>

InceptionV3 shows better results then vgg19 and mobilenet.

### Hyperparameter tuning on InceptionV3

There are various hyperparameters like 
1. Learning rate.
2. Optimizer.
3. Number of hidden units.
4. Number of hidden layers.
5. Batch size.
6. Number of epochs etc.

But I have selected above two hyperparameters for tuning. Let's look at the tensorboard's hparams visualization. 
<img src="https://github.com/dhairyasheelsutar/Food-11-image-classification/blob/master/imgs/hp1.PNG"  alt="Tensorboard hparams visualization"/>

It appears that Adam optimizer with learning rate 0.001 shows best results.

### Error analysis

The error analysis technique gives the insight regarding where the model is mistaking. Here is the error contribution per class in the overall error.

<img src="https://github.com/dhairyasheelsutar/Food-11-image-classification/blob/master/imgs/error_analysis_bar.png" alt="Error analysis" />

Most of the contribution is due to the Egg class.




