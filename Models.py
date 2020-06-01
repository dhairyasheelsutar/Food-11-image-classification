#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


class ModelTrain:

    def __init__(self, model, target_dim, batch_size):
        self.model = model
        self.target_dim = target_dim
        self.batch_size = batch_size
     
    def freeze_layers(self, no_layers):
        
        for layers in self.model.layers[:-no_layers]:
            layers.trainable = False

        for layers in self.model.layers[-no_layers:]:
            layers.trainable = True 
        
    def train(self, train_generator, validation_generator, callbacks = [], epochs = 50):
        
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size,
            callbacks=callbacks,
            epochs=epochs
        )
 

# In[3]:


class TrainingCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if logs['acc'] > 0.90:
            print("Reached target training accuracy")
            self.model.stop_training = True


# In[ ]:




