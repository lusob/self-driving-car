# lusob SDC (Open Source Self-Driving Car)

Self driving car model trainer for Udacity [Challenge #2](https://medium.com/udacity/challenge-2-using-deep-learning-to-predict-steering-angles-f42004a36ff3#.5650j9v4s)
Once trained using just frame pictures files named with timestamp, it will predict the steering angle for every specific frame

##### Intall dependencies
You need to install keras and optionally Tensorflow as backend

Fo ubuntu run these commands:
sudo apt-get -y -qq install python-pip python-dev python-scipy
sudo pip install keras

##### To train the model
./sdc.py --dataset path_to_data/steering_angles.csv --imgs  path_to_data/imgs

##### To evaluate the model and generate a final csv with the tests dataset
./sdc.py --evaluate --imgs path_to_data/test_imgs

