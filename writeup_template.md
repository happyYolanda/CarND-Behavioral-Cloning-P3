#Behavioral Cloning

##Writeup Report

**Behavioral Cloning Project**

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to predict the steering angle of the simulator using CNN.

My first step was to use a convolution neural network model similar to the 'classifying traffic signs' project. I found pooling layers may be counterproductive for keeping a car centered on the road, so I avoided the use of pooling layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. To combat the overfitting, I flip the image and use dropout.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track: in the first right corner and on the bright. To improve the driving behavior in these cases, I trained the simulator to drive on the opposite direction and and randomly crop a frame out of the image.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py) consisted of a convolution neural network with the following layers and layer sizes:
1) preprocessing layer that takes in images of shape 64x64x3. 2) two convolutional with kernels of size k=(8,8), stride of s=(4,4) and 32 and 64 channels repectively. 3) convolutional layer uses k=(4,4) kernels, a stride of s=(2,2) and 128 channels. 4) convolutional layer use k=(2,2), a stride s=(1,1) and 128 channels. 5) two fully connected layers with ReLU activations as well as dropout regularization. 6) a single neuron that provides the predicted steering angle. ReLU activations are used throughout the whole network.  

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

| Layer (type)                    | Output Shape      |    Param #   |  Connected to                     
|---------------------------------|-------------------|--------------|------------------------- 
| lambda_1 (Lambda)               | (None, 64, 64, 3) |    0          | lambda_input_1[0][0]             
| convolution2d_1 (Convolution2D) | (None, 16, 16, 32)  |  6176        | lambda_1[0][0]                   
| activation_1 (Activation)     |   (None, 16, 16, 32)   | 0           | convolution2d_1[0][0]            
| convolution2d_2 (Convolution2D) |  (None, 4, 4, 64)     | 131136    |  activation_1[0][0]               
| relu2 (Activation)              | (None, 4, 4, 64)     | 0          | convolution2d_2[0][0]            
| convolution2d_3 (Convolution2D) | (None, 2, 2, 128)    | 131200     | relu2[0][0]                      
| activation_2 (Activation)       | (None, 2, 2, 128)    | 0          | convolution2d_3[0][0]            
| convolution2d_4 (Convolution2D) | (None, 2, 2, 128)    | 65664      | activation_2[0][0]               
| activation_3 (Activation)       | (None, 2, 2, 128)    | 0          | convolution2d_4[0][0]            
| flatten_1 (Flatten)          |    (None, 512)          | 0          | activation_3[0][0]               
| dropout_1 (Dropout)           |   (None, 512)          | 0          | flatten_1[0][0]                  
| dense_1 (Dense)                |  (None, 128)          | 65664      | dropout_1[0][0]                  
| activation_4 (Activation)       | (None, 128)         |  0          | dense_1[0][0]                    
| dropout_2 (Dropout)             | (None, 128)        |   0          | activation_4[0][0]               
| dense_2 (Dense)                |  (None, 128)       |    16512      | dropout_2[0][0]                  
| dense_3 (Dense)                |  (None, 1)        |     129        | dense_2[0][0]       

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back from the side of the road. 

To augment the data sat, I also flipped images and angles thinking that this would learn both right and left turns. For example, here is an image that has then been flipped:

![flipped image][img_flip.jpg]

In order to simulate a bending road, the image is sheared horizontally:
![sheared image][img_shear.jpg]

I also randomly crop subsections of the image to simulate the car being offset from the middle of the road:
![cropped image][img_crop.jpg]

At last, I use change the brightness of image to simulate differnt lighting conditions:
![brightness image][img_bright.jpg]

The final training images are then generated in batches of 20 on the fly with 2000 images per epoch. A python generator creates new training batches by applying the aforementioned transformations with accordingly corrected steering angles. 

####4. Model Training
After the collection process, I had 23211 number of data points.I finally randomly shuffled the data set and put 10% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
