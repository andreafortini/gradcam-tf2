# GRADCAM

Implementation of the Grad-CAM algorithm in an easy-to-use class, optimized for transfer learning projects and written using Keras and Tensorflow 2.x.

## Requirement

 - Python 3.6.x
 - Matplotlib 3.4.x
 - Numpy 1.19.x
 - Tensorflow 2.4.x

## Usage
Please take a look of examples folder for a complete example.

```python
from gradcam import Gradcam

gc = Gradcam(model, 
             layer_name="top_conv",
             img_path=img_path,
             size=img_size,
             inner_model=model.get_layer("efficientnetb0"))

gc.generate_stack_img(save_name="../output/example_out")
```
![output image](https://github.com/andreafortini/gradcam-tf2/blob/master/output/example_out.png?raw=true)

## Reference
[[1]](https://arxiv.org/abs/1610.02391) Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

