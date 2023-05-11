<h1 align="center">A Neural Algorithm of Artistic Style</h1>
PyTorch implementation of "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.


## Methods
Instead of optimizing model weights, the approach described in [1] optimizes pixel values in the input image, which means this neural
transfer algorithm works on images of arbitrary size. 

The `NeuralStyleTransfer` model is built directly from the layers in PyTorch's pretrained `vgg19` model. 
However, its `MaxPool2d` layers are replaced with `AvgPool2d` layers because average-pooling was found to produce better results in [1]. 
Furthermore, after each layer specified in `CONTENT_LAYERS` and `STYLE_LAYERS`, an appropriate loss probe 
(either `ContentLoss` or `StyleLoss`) was inserted to calculate the loss of the feature maps produced by those layers. 
During training, the total content and style losses from these layers are weighted by `CONTENT_WEIGHT` and `STYLE_WEIGHT` respectively to calculate the total loss.

Before training, the given style image is resized to be the same shape as the content image so that their corresponding feature maps have the same dimensions and their Gram matrices can be computed correctly.
Also, gradient calculations for the model weights are turned off since those do not need to be tuned. 
Following [1], the input image starts as a white-noise tensor and is optimized during training to become the results shown below.


## Results

<div align="center">
  <table>
    <tr>
      <th align="center">
        Content Images →
        <br><br>
        Style Images ↓
      </th>
      <td><img src="input/content/ballerina.jpg" width="300px"></td>
      <td><img src="input/content/tubingen.jpg" width="300px"></td>
    </tr>
    <tr>
      <td><img src="input/style/picasso.jpg" width="300px"></td>
      <td><img src="output/05_10_2023/13_48_08_lbfgs/ballerina-picasso.png" width="300px"></td>
      <td><img src="output/05_10_2023/14_49_57/tubingen-picasso.png" width="300px"></td>
    </tr>
    <tr>
      <td><img src="input/style/starry_night.jpg" width="300px"></td>
      <td><img src="output/05_10_2023/15_46_12/ballerina-starry_night.png" width="300px"></td>
      <td><img src="output/05_10_2023/14_12_06/tubingen-starry_night.png" width="300px"></td>
    </tr>
    <tr>
      <td><img src="input/style/shipwreck.jpg" width="300px"></td>
      <td><img src="output/05_10_2023/18_00_33/ballerina-shipwreck.png" width="300px"></td>
      <td><img src="output/05_10_2023/18_04_11_loss_after_relu/tubingen-shipwreck.png" width="300px"></td>
    </tr>
  </table>
</div>

# Notes
- lbfgs much better than adam
- larger images have much finer textures, perhaps due to size of kernels staying constant between small vs large images?
- style image has a big impact on whether the training is stable or not
  - picasso.jpg is very stable but starry_night.jpg is very unstable
