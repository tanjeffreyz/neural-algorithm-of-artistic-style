<h1 align="center">A Neural Algorithm of Artistic Style</h1>
PyTorch implementation of "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.



<table>
  <tr>
    <td></td>
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
</table>

- lbfgs much better than adam
- larger images have much finer textures, perhaps due to size of kernels staying constant between small vs large images?
- style image has a big impact on whether the training is stable or not
  - picasso.jpg is very stable but starry_night.jpg is very unstable
