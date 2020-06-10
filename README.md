# Food 11 image classification

This is image classification on food 11 dataset. The dataset is taken from <a href='https://www.kaggle.com/vermaavi/food11'>https://www.kaggle.com/vermaavi/food11</a>. The dataset contains 16643 food images categorized into 11 foods. The 11 categories are Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. 

## Approach

<ul>
  <li>
    Train 3 pretrained models like vgg19, inceptionV3 and mobilenet on this dataset to find out which model works the best. 
  </li>
  <li>
    Select the model with highest accuracy on test set.  
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
