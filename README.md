# Pixel Prediction

Predict pixels on handwritten digit (MNIST dataset) with RNN (GRU cell)

The dimension for each image is 28 x 28 (784 pixels)

<br><br/>
There are two parts in this project:

<dl>
  <dt> Part A:</dt>
  <ul>
  <li>Architecture: input (1 pixel) &rarr; GRU cell (128 units) &rarr; linear layer + sigmoid</li>
  <li>At each time <i>t</i>, the model receives as input a pixel value <i>x<sub>t</sub></i> and tries to predict the next pixel in the images <i>x<sub>t+1</sub></i> based on the current input and the recurrent state.</li>
  </ul>
  
  <dt> Part B:</dt>
  <li>Mask the last 300 pixels on each images and use the saved model from Part A to predict the masked pixels.</li>
  <li>No training involved in this part.</li>
</dl>

<dt>Result (Part B):</dt>
<p align="center"> 
<img src="https://github.com/ykteh93/Deep_Reinforcement_Learning-Atari/blob/master/MsPacman/Graphs_and_Figure/Plot%20of%20Loss%20Over%201%20million%20Steps.png">
</p>
