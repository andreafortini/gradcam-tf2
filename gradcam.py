from tensorflow import keras
from PIL import Image

import os
import random
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg


class Gradcam:
    """
    Class to rapresent heatmap of output of neural network based on Grad-CAM algorithm.
    
    ...

    Attributes
    __________
    model: HD5
        model to analyze
    layer_name: str
        name of the layer of the model to analyze
    img_path: str
        Path of the image
    size: (int, int)
        size of the image that the model takes as input 
    pred_index (opt): int
        index of target label of image
    inner_model (opt): str
        name of the inner model, used for transfer learning

    Methods
    _______
    get_img_array():
        Load an image and convert it into numpy array.
    set_index_from_path():
        Predict the label of the image passed.
    make_gradcam_heatmap(img_array):
        Calculate the heatmap.
    overlay_heatmap(heatmap, campath, alpha, cmap):
        Superimposed original image and heatmap and return another image.
    generate_stack_img(figsize, save_name, superimposed_name, alpha, cmap):
        Mix together original image, heatmap and superimposed image and save them
        into a single compose image.
    
    """

    def __init__(self, model, layer_name, img_path, size, pred_index=None, inner_model=None):
        """Gradcam constructor to inizialize the object
        
        Parameters
        __________
        model: HD5
            model to analyze
        layer_name: str
            name of the layer of the model to analyze
        img_path: str
            Path of the image
        size: (int, int)
            size of the image that the model takes as input 
        pred_index (opt): int
            index of target label of image
        inner_model (opt): str
            name of the inner model, used for transfer learning
        """

        self.model = model
        self.layer_name = layer_name
        self.img_path = img_path
        self.size = size
        self.pred_index = pred_index
        self.inner_model = inner_model
        
        if self.inner_model == None:
            self.inner_model = model

        if self.pred_index == None:
            self.pred_index = self.set_index_from_path()

    def get_img_array(self):
        """Load an image and convert it into numpy array.

        Returns
        _______
        (ndarray) = Image converted into array
        """

        img = keras.preprocessing.image.load_img(self.img_path, target_size=self.size)
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)

        return array

    def set_index_from_path(self):
        """Predict the label of the image passed.

        Returns
        _______
        (int) = Predicted index
        """

        array = self.get_img_array()
        self.model.layers[-1].activation = None
        preds = self.model.predict(array)
        i = np.argmax(preds, axis=1)

        self.pred_index = i[0]

        return self.pred_index
        
    def make_gradcam_heatmap(self, img_array):
        """Calculate the heatmap.

        Paramenters
        ___________
        img_array: ndarray
            Image converted into array

        Returns
        _______
        (ndarray) = Heatmap of image converted into array
        """

        grad_model = tf.keras.models.Model(
            inputs=[self.inner_model.inputs],
            outputs=[self.inner_model.get_layer(self.layer_name).output,
                     self.inner_model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            last_conv_layer_output, preds = grad_model(inputs)
            class_channel = preds[:, self.pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, cam_path, alpha, cmap):
        """Superimposed original image and heatmap and return another image.

        Parameters
        __________
        heatmap: ndarray
            Heatmap of the image
        cam_path: str
            Path where to save only superimposed image generated
        alpha: float
            Parameter used as weight for color map when overlapping is applied
        cmap: str
            Type of color map used for heatmap

        Returns
        _______
        (Image) = Image of superimposed image of original and heatmap
        """

        img = keras.preprocessing.image.load_img(self.img_path, target_size=self.size)
        img = keras.preprocessing.image.img_to_array(img)

        heatmap = np.uint8(255 * heatmap)

        color_map = cm.get_cmap(cmap)
        color_map = color_map(np.arange(256))[:, :3]
        color_map = color_map[heatmap]
        color_map = keras.preprocessing.image.array_to_img(color_map)
        color_map = color_map.resize((img.shape[1], img.shape[0]))
        color_map = keras.preprocessing.image.img_to_array(color_map)

        # Superimpose the heatmap on original image
        superimposed_img = color_map * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        if cam_path != None:
            superimposed_img.save(cam_path)

        return superimposed_img
 
    def generate_stack_img(self, figsize=(17,9), save_name=None, superimposed_name=None, alpha=0.4, cmap="jet"):
        """Mix together original image, heatmap and superimposed image and save them
        into a single compose image.

        Parameters
        __________
        figsize (opt): (int, int)
            Figsize of final image
            default=(17,9)
        save_name (opt): str
            Name of image to save.
            default={original_image_name}_gradcam_{current_time} 
        superimposed_name (opt): str
            Name of superimposed image to save.
            default=None, superimposed image will not be saved
        alpha (opt): float
            Parameter used as weight for color map when overlapping is applied
            default=0.4
        cmap (opt): str
            Type of color map used for heatmap
            default="jet"

        Returns
        _______
        None
        """

        img_array = self.get_img_array()

        heatmap = self.make_gradcam_heatmap(img_array)
        superimposed_gradcam = self.overlay_heatmap(heatmap, cam_path=superimposed_name, alpha=alpha, cmap=cmap)

        if save_name == None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            extract_name = (self.img_path.split("/")[-1]).rsplit(".",1)[0]
            save_name = extract_name + "_gradcam_" + current_time 

        img_original = Image.open(self.img_path)
        img_original = img_original.resize(self.size)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("Grad-CAM | Prediction: " + str(self.pred_index))
        ax1.imshow(img_original)
        ax2.matshow(heatmap)
        ax3.matshow(superimposed_gradcam)
        fig.savefig(save_name)

