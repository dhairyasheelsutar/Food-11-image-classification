#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tempfile
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


def add_regularization(model, layers, regularizer=tf.keras.regularizers.l2(0.001)):
    for layer in layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)
    
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


# In[3]:


def get_data_generators(datagen, target_dim = 224, batch_size = 32):
    
    train_generator = datagen.flow_from_directory(
        directory='./training/', 
        target_size=(target_dim, target_dim),
        batch_size=batch_size,
    )

    validation_generator = datagen.flow_from_directory(
        directory='./validation/',
        target_size=(target_dim, target_dim),
        batch_size=batch_size
    )
    
    return train_generator, validation_generator


# In[ ]:

def plot_model_results(num_epochs, history, key):

    plt.plot(range(num_epochs), history[key])
    plt.plot(range(num_epochs), history['val_' + key])
    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.show()




