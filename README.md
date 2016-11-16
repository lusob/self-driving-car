# Open Source Self-Driving Car

Self driving car model trainer for Udacity [Challenge #2](https://medium.com/udacity/challenge-2-using-deep-learning-to-predict-steering-angles-f42004a36ff3#.5650j9v4s)

Once trained using just frame pictures files named with timestamp, it will predict the steering angle for every specific frame

##### Intall dependencies
You need to install keras and optionally Tensorflow as backend

Fo ubuntu run these commands:
```bash
sudo apt-get -y -qq install python-pip python-dev python-scipy python-h5py
sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
sudo pip install keras
```

##### Get the data
You can get the udacity driving data from this torrent

http://academictorrents.com/details/9b0c6c1044633d076b0f73dc312aa34433a25c56

It's a final dataset that consists of PNG images from the center camera only along with additional training data (all three cameras and steering angles in CSV format) is available here.
This is a large file, around 70GB

Once the model is trained to evaluate ant test it you can download this smaller set of images (1.9GB compressed) 

http://bit.ly/2eVhdrA

##### Train the model
```bash
./sdc.py --dataset path_to_data/steering_angles.csv --imgs  path_to_data/imgs
```

##### Evaluate the model and generate a final csv with the predicted angles for the tests dataset

```bash
./sdc.py --evaluate --imgs path_to_data/test_imgs
```

![alt tag](https://github.com/lusob/self-driving-car/raw/master/data-viewer-screenshot.png)

