#!/usr/bin/env python
# Author: lusob (https://github.com/lusob)
#
# To train the model:
# ./sdc.py --dataset path_to_data/steering_angles.csv --imgs  path_to_data/imgs
#
# To predict, generating a final csv with the calculated angles
# ./sdc.py --predict imgs  path_to_data/imgs
#
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.preprocessing.image import ImageDataGenerator
import csv
import numpy as np
from PIL import Image

os.environ["KERAS_BACKEND"] = "tensorflow"

class ImgInfo():
    def __init__(self, imgs_path, imgs_format):
        self.imgs_path = imgs_path
        self.imgs_format = imgs_format
        self.ch, self.rows, self.cols = \
            get_imgs_res(self.imgs_path, self.imgs_format)

def read_data(dataset, epoch_size, img_info):
        images = np.empty((epoch_size,img_info.rows,img_info.cols,img_info.ch),dtype="float32")
        angles = np.empty((epoch_size),dtype="float32")
        with open(dataset, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
            i = 0
            for row in reader:
                if row[0] == 'timestamp':
                    continue
                ts = int(row[0])
                #img = '../bag2csv_extractor/data/center/%s.pgm' % ts
                img = '%s%s.%s' % (img_info.imgs_path, ts, img_info.imgs_format)
                if not os.path.exists(img):
                    continue
                img = Image.open(img)
                images[i] = np.array(img)
                angles[i] = float(row[1])
                i += 1
                if i == epoch_size:
                    break
        datagen = ImageDataGenerator()
        return datagen.flow(images, angles, batch_size=256)

def read_data_test(img_info):
        img_list = sorted(os.listdir(img_info.imgs_path))
        images = np.empty((len(img_list),img_info.rows,img_info.cols,img_info.ch),dtype="float32")
        angles = np.empty((len(img_list)),dtype="float32")
        time_stamps = []
        i = 0
        for filename in img_list:
            if filename.endswith('.%s' % img_info.imgs_format):
                # print(os.path.join(directory, filename))
                img = os.path.join(img_info.imgs_path, filename)
                img = Image.open(img)
                images[i] = np.array(img)
                ts = os.path.basename(filename).split('.')[0]
                time_stamps.append(ts)
                i += 1
        datagen = ImageDataGenerator()
        return time_stamps, len(img_list), datagen.flow(images, angles, batch_size=256)

def get_imgs_res(imgs_path, imgs_format):
    filename = os.listdir(imgs_path)[0]
    if filename.endswith('.%s' % imgs_format):
        # print(os.path.join(directory, filename))
        img = os.path.join(imgs_path, filename)
        img = Image.open(img)
        width, height = img.size
        return 3, height, width
    else:
        raise Exception("Format file incorrect, set the imgsformat param")

def create_model(img_info):
    # Set dimensions of the network based on image resolution
    model = Sequential()
    # Regulerize data
    model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(img_info.rows, img_info.cols, img_info.ch),
        output_shape=(img_info.rows, img_info.cols, img_info.ch)))

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=['binary_accuracy'])

    print('Model is created and compiled..')
    return model

def predict(args):
    img_info = ImgInfo(args.imgs, args.imgsformat)
    # create model and load weights
    model = create_model(img_info)
    model.load_weights("./outputs/steering_model/steering_angle.keras")
    print("Loaded model from disk")
    time_stamps, val_samples, data_gen = read_data_test(img_info)
    score = model.predict_generator(data_gen, val_samples)
    with open('udacity_test_angles.csv','wb') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['frame_id','steering_angle'])
        for i in xrange(len(score)):
            predicted_angle = score[i][0]
            ts = time_stamps[i]
            csv_out.writerow((ts,predicted_angle))
            i += 1
    print "Prediction done and csv file generated"

def train_model(args):
    dataset = args.dataset
    epoch_size = args.epochsize
    epoch = args.epoch
    img_info = ImgInfo(args.imgs, args.imgsformat)
    model = create_model(img_info)
    if os.path.exists("./outputs/steering_model"):
        print("Loading model weights.")
        model.load_weights("./outputs/steering_model/steering_angle.keras", True)
    model.fit_generator(read_data(dataset, epoch_size, img_info),
        nb_epoch=epoch,
        samples_per_epoch=epoch_size,
    )

    print("Saving model weights and configuration file.")
    if not os.path.exists("./outputs/steering_model"):
        os.makedirs("./outputs/steering_model")

    model.save_weights("./outputs/steering_model/steering_angle.keras", True)
    with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

if __name__ == "__main__":
    # Args parsing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument('--dataset', type=str, help='Driving log file', required=True)
    parser_train.add_argument('--imgs', type=str, help='Driving frames', required=True)
    parser_train.add_argument('--imgsformat', type=str, default='png', help='Driving frames file format')
    parser_train.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    parser_train.add_argument('--epochsize', type=int, default=15211, help='How many frames per epoch.')
    parser_train.set_defaults(func=train_model)

    parser_eval = subparsers.add_parser('predict', help='Use the trained model against test images to predict the angles, it will generate a final csv with timestamp and angles')
    parser_eval.add_argument('--imgs', type=str, help='Driving frames to predict steering angle from', required=True)
    parser_eval.add_argument('--imgsformat', type=str, default='png', help='Driving frames file format')
    parser_eval.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)
