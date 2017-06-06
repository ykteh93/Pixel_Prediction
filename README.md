# Pixel Prediction

Predict pixels on handwritten digit (MNIST dataset) with RNN (GRU cell)

The dimension for each image is 28 x 28 (784 pixels)

<br><br/>
There are two parts in this project:

<dl>
  <dt> Part A:</dt>
  <ul>
  <li>Architecture: input &rarr; GRU cell (128 units) &rarr; linear layer + sigmoid</li>
  <li>At each time <i>t</i>, the model receives as input a pixel value <i>x<sub>t</sub></i> and tries to predict the next pixel in the images <i>x<sub>t+1</sub></i> based on the current input and the recurrent state.</li>
  </ul>
  
  <dt> Part B:</dt>
  <ul>
  <li>Sample 100 images from test set and mask/remove the last 300 pixels in each image.</li>
  <li>Use the saved model from Part A to predict the masked pixels.</li>
  </ul>
</dl>

<br><br/>
<dl>
<dt>Result (Part B):</dt>
</dl>
<p align="center"> 
<img src="https://github.com/ykteh93/Pixel_Prediction/blob/master/Part%20B/image/For_README.png" style="width: 50%; height: 50%"/>
</p>
