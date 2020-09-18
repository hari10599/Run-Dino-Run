from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import cv2
import numpy as np
from io import BytesIO
import base64
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from collections import deque
import os
import random
import pickle
import bz2
import copy
from GameDriver import GameDriver
import config

class Game:
    def __init__(self):
        self.webdriver = webdriver.Chrome(executable_path = '/Users/harikrishna/Documents/Projects/Run Dino Run/chromedriver')
        self.webdriver.set_window_position(x=-10, y=0)
        try:
            self.webdriver.get('chrome://dino')
        except:
            print()
        time.sleep(0.5)
        self.webdriver.execute_script("Runner.config.ACCELERATION=0")
        self.webdriver.execute_script("document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")
    def is_crashed(self):
        return self.webdriver.execute_script("return Runner.instance_.crashed")
    def restart(self):
        self.webdriver.execute_script("Runner.instance_.restart()")
    def jump(self):
        self.webdriver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

def get_state(game, action):
    reward = 1
    if action == 1:
        game.jump()
    image = get_screenshot(game.webdriver)
    isOver = game.is_crashed()
    if isOver:
        reward = -10
    return image, reward, isOver


def get_screenshot(webdriver):
    image_b64 = webdriver.execute_script("canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)")
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    image = image[:300, :700]
    image = cv2.inRange(image, 70, 90)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.resize(image, (config.DIMENSION, config.DIMENSION))

def buildModel():
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same',strides=(4, 4),input_shape=(config.DIMENSION, config.DIMENSION, config.FRAMES)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(config.ACTIONS))
    adam = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=adam)

    if not os.path.isfile('/Users/harikrishna/Documents/Projects/Run Dino Run/model.h5'):
        model.save_weights("model.h5")
    return model

def training(game, model):
    image, reward, over = get_state(game, 0)
    stacked = np.stack((image, image, image, image), axis = 2)
    stacked = stacked.reshape(1, *stacked.shape)
    initial = copy.deepcopy(stacked)

    sfile = bz2.BZ2File('epsilon', 'r')
    epsilon = pickle.load(sfile)
    sfile.close()

    sfile = bz2.BZ2File('replay', 'r')
    deque = pickle.load(sfile)
    sfile.close()

    sfile = bz2.BZ2File('framesCount', 'r')
    framesCount = pickle.load(sfile)
    sfile.close()

    model.load_weights("model.h5")
    adam = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=adam)

    while True:
        action = 0
        if random.random() <= epsilon:
            action = random.randrange(config.ACTIONS)
        else:
            predicted = model.predict(stacked)
            print(predicted)
            if predicted[0][1] >= predicted[0][0]:
                action = 1
        if action == 0:
            print("Forward")
        else:
            print("Jump")
        image, reward, isOver = get_state(game, action)
        #time.sleep(0.1)
        image = image.reshape(1, image.shape[0], image.shape[1], 1)
        newStacked = np.append(image, stacked[:, :, :, :3], axis=3)

        if epsilon > config.FINAL_EPSILON and framesCount > 3000:
            epsilon-=(config.INITIAL_EPSILON - config.FINAL_EPSILON) / 10000

        deque.append((stacked, action, reward, newStacked, isOver))
        if len(deque) > config.REPLAY_MEMORY:
            deque.popleft()

        if framesCount > 3000:
            batch = random.sample(deque, 32)
            inputs = np.zeros((32, stacked.shape[1], stacked.shape[2], stacked.shape[3]))
            outputs = np.zeros((inputs.shape[0], config.ACTIONS))
            for i in range(32):
                state1 = batch[i][0]
                action = batch[i][1]
                reward = batch[i][2]
                state2 = batch[i][3]
                over = batch[i][4]

                inputs[i:i+1] = state1
                outputs[i] = model.predict(state1)
                qNext = model.predict(state2)

                if over:
                    outputs[i, action] = reward
                else:
                    outputs[i, action] = reward + 0.99 * np.max(qNext)
            model.train_on_batch(inputs, outputs)

        if isOver:
            stacked = initial
            game.jump()
        else:
            stacked = newStacked
        framesCount+=1

        if framesCount%100 == 0:
            model.save_weights("model.h5", overwrite=True)
            sfile = bz2.BZ2File('replay', 'w')
            pickle.dump(deque, sfile)
            sfile.close()
            sfile = bz2.BZ2File('framesCount', 'w')
            pickle.dump(framesCount, sfile)
            sfile.close()
            sfile = bz2.BZ2File('epsilon', 'w')
            pickle.dump(epsilon, sfile)
            sfile.close()

            print(framesCount, " " , epsilon)


if not os.path.isfile('/Users/harikrishna/Documents/Projects/Run Dino Run/epsilon'):
    sfile = bz2.BZ2File('epsilon', 'w')
    pickle.dump(config.INITIAL_EPSILON, sfile)
    sfile.close()

if not os.path.isfile('/Users/harikrishna/Documents/Projects/Run Dino Run/replay'):
    sfile = bz2.BZ2File('replay', 'w')
    d = deque()
    pickle.dump(d, sfile)
    sfile.close()

if not os.path.isfile('/Users/harikrishna/Documents/Projects/Run Dino Run/framesCount'):
    sfile = bz2.BZ2File('framesCount', 'w')
    t = 0
    pickle.dump(t, sfile)
    sfile.close()

game = Game()
game.jump()
model = buildModel()
training(game, model)