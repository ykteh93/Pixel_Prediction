# Pixel Prediction

Predict pixels on handwritten digit (MNIST dataset) with RNN (GRU cell)

<br><br/>
There are two parts in this project:

<dl>
  <dt> Part A:</dt>
  <dd> At each time <i>t</i>, the model receives as input a pixel value <i>x<sub>t</sub></i> and tries to predict the next pixel in the images <i>x<sub>t+1</sub></i> based on the current input and the recurrent state.</dd>
  
   <dt> Part B:</dt>
  <dd>Mask 300 pixels on each image and use the saved model from Part A to predict all 300 pixels.</dd>
</dl>

Result (Part B):
![Result](https://www.dropbox.com/s/zc4vujvksonwgc3/Example.png)


