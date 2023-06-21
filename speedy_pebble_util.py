import sys
import os
import warnings
import random
import cv2
import numpy as np
import torchvision
import torchvision.transforms as VT
import torch
import matplotlib.pyplot as plt
import math
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class Pebble():
    def __init__(self, location, number, count, startTime):
        self.actualLocations = [location]
        self.number = number
        self.firstSeen = count
        self.lastSeen = count
        self.startTime = startTime
        self.lastSeenTime = startTime
        self.digits = np.zeros((3, 10))
        self.currentPebbleBox = None
        self.currentDigitBoxes = None
        self.isConverged = False
        self.ConvergedClassification = '???'

        print('Pebble #'+str(self.number)+' has been created')

    def addLocation(self, location, frameNumber, videoTime):
        self.actualLocations.append(location)
        self.lastSeen = frameNumber
        self.lastSeenTime = videoTime
        # print('Pebble #'+str(self.number) +
        #       ' has been seen in frame '+str(self.lastSeen))

    def addDigits(self, labels, scores):
        for l in range(len(labels)):
            # only add if score is greater than 0.8
            if scores[l] >= 0.8:
                self.digits[l][labels[l]] += 1
            if scores[l] >= 0.98:
                self.digits[l][labels[l]] += 1
            # self.digits[l][labels[l]] += scores[l]
        print('Pebble #'+str(self.number) +
              ' has updated digits '+str(self.digits))

    def addDigitBoxes(self, boxes):
        self.currentDigitBoxes = boxes
        self.currentPebbleBox = boxes[0]

    def resetBoxes(self):
        self.currentPebbleBox = None
        self.currentDigitBoxes = None

    def obtainFinalClassification(self):
        if self.isConverged:
            return self.ConvergedClassification
        # check if no good prediction
        if np.sum(self.digits) == 0:
            return '???'
        # obtain top prediction for each digit position
        classification = ''
        converged = True
        for d in range(len(self.digits)):
            # take argmax of each position
            maxPos = np.argmax(self.digits[d])
            if self.digits[d][maxPos] == 0:
                # no maximum for this position
                return '???'
            elif self.digits[d][maxPos] < 10:
                converged = False
            classification += str(maxPos)

        # check if digits have converged
        if converged:
            self.isConverged = True
            self.ConvergedClassification = classification
        return classification


def updatePebbleLocation(pebbleBox, pebbles, distThreshold, numOfPebbles, frameNumber, videoTime):
    # calculate midpoint of box
    x_center = int((pebbleBox[0]+pebbleBox[2])/2)
    y_center = int((pebbleBox[1]+pebbleBox[3])/2)

    # calculate distance from this midpoint to previous pebble locations
    for i in range(len(pebbles)):
        pebbleLocation = pebbles[i].actualLocations
        # obtain last midpoint of this pebble
        x_last = pebbleLocation[-1][0]
        y_last = pebbleLocation[-1][1]

        # calculate distance
        dist = math.sqrt(
            (math.pow((x_center-x_last), 2)+math.pow((y_center-y_last), 2)))
        if dist < distThreshold:
            # active pebble found, add location
            pebbles[i].addLocation((x_center, y_center),
                                   frameNumber, videoTime)
            return pebbles[i], pebbles, numOfPebbles
    # if not found, need to create new pebble
    numOfPebbles += 1
    newPebble = Pebble((x_center, y_center), numOfPebbles,
                       frameNumber, videoTime)
    # add to list
    pebbles.append(newPebble)
    # return new pebble
    return pebbles[-1], pebbles, numOfPebbles
