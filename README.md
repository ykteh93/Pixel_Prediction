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
<br><br/>
<p align="center"> 
<img src="https://github.com/ykteh93/Deep_Reinforcement_Learning-Atari/blob/master/MsPacman/Graphs_and_Figure/Plot%20of%20Loss%20Over%201%20million%20Steps.png">
</p>


