import sys
import os
import warnings
import random
import cv2
import numpy as np
import torchvision
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
from PIL import Image
import math

warnings.filterwarnings('ignore')

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# set to evaluation mode
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

digit_detection_model = torch.load(
    r'./saved_models/idvl_digit_detector_10.pkl')
digit_detection_model.to(device)
digit_detection_model.eval()


def no_overlap_pred(boxes, labels, scores, maxDim):
    minDistThresh = maxDim * 0.1
    # maxDistThresh = maxDim * 0.5
    # first get midpoint of all boxes
    midpoints = []
    for box in boxes:
        x_mid = (box[0] + box[2])/2
        y_mid = (box[1] + box[3])/2
        midpoints.append((x_mid, y_mid))
    # now select top three that are not close
    # we can always include top prediction
    indexesToUse = [0]
    for d in range(1, len(boxes)):
        # check distance between current index and other indexes in list
        usable = True
        for indexUsed in indexesToUse:
            # compute distance
            m1 = midpoints[indexUsed]
            m2 = midpoints[d]
            dist = math.sqrt(
                (math.pow((m1[0]-m2[0]), 2)+math.pow((m1[1]-m2[1]), 2)))
            # threshold dist
            if dist < minDistThresh:
                # too close or too far
                usable = False
        # if not skipped add to our indexes to use
        if usable and scores[d] >= 0.7:
            indexesToUse.append(d)
        # if we reached three can finish
        if len(indexesToUse) == 3:
            break

    # update lists
    boxes = boxes[indexesToUse]
    labels = labels[indexesToUse]
    scores = scores[indexesToUse]

    return boxes, labels, scores


def get_number_from_pred(boxes, labels, scores, bottomY):
    # check if length is greater than three and if so select three most distinct
    if len(labels) >= 3:
        boxes, labels, scores = no_overlap_pred(boxes, labels, scores)
    # if only 2 or less predictions, not usable
    if len(labels) < 3:
        return None, None, None, None
    # print('boxes:', boxes)

    # get indices to sort boxes by minimum x value
    sortInd = np.argsort(boxes[:, 0])
    # put labels in order
    labels = labels[sortInd]
    boxes = boxes[sortInd]
    scores = scores[sortInd]

    # check if this image is flipped
    # check lowest point of detections
    lowestY = 0
    for box in boxes:
        minOfBox = max(box[1], box[3]).item()
        if minOfBox > lowestY:
            lowestY = minOfBox
    if abs(bottomY-lowestY) > 40:
        # not aligned and is flipped
        # print('Flipped image found bottomY and lowestY are:', bottomY, lowestY)
        return None, None, None, None
    # print('bottomY and lowestY are:', bottomY, lowestY)

    # create number
    number = ''
    # iterate through each label adding it to our number
    for i in range(len(labels)):
        label = str(labels[i].item())
        # check if 10 and covert to 0
        if label == '10':
            label = '0'
            labels[i] = 0
        number += label

    return number, boxes, labels, scores


def normalize(arr):
    """
    Linear normalization
    normalize the input array value into [0, 1]
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # print("...arr shape", arr.shape)
    # print("arr shape: ", arr.shape)
    for i in range(3):
        minval = arr[i, :, :].min()
        maxval = arr[i, :, :].max()
        if minval != maxval:
            arr[i, :, :] -= minval
            arr[i, :, :] /= (maxval-minval)
    return arr


def fig_num(img, number):
    # put number in bottom left corner of image
    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get boundary of this text
    textsize = cv2.getTextSize(number, font, 3, 6)[0]

    # get coords based on boundary
    textX = int((img.shape[1] - textsize[0]) / 2)
    textY = int((img.shape[0] + textsize[1]) / 2)
    cv2.putText(img, number, (textX, img.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 255, 255), thickness=6)


def fig_draw(img, box, label, score):
    # draw predicted bounding box and class label on the input image
    # draw predicted bounding box and class label on the input image
    xmin = round(box[0])
    ymin = round(box[1])
    xmax = round(box[2])
    ymax = round(box[3])

    predText = '' + str(label) + ':' + str(int(score*100)/100).lstrip('0')

    if label == 0:  # start with background as 0
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (181, 252, 131), thickness=3)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (181, 252, 131), thickness=3)
    elif label == 1:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 255, 0), thickness=3)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), thickness=3)
    elif label == 2:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 0, 255), thickness=3)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), thickness=3)
    elif label == 3:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 100, 255), thickness=3)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 100, 255), thickness=3)
    elif label == 4:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 142, 142), thickness=3)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 142, 142), thickness=3)
    elif label == 5:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 0, 255), thickness=3)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 255), thickness=3)
    elif label == 6:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 255, 255), thickness=3)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 255), thickness=3)
    elif label == 7:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255, 255, 0), thickness=3)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 0), thickness=3)
    elif label == 8:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (121, 252, 206), thickness=3)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (121, 252, 206), thickness=3)
    elif label == 9:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (119, 118, 193), thickness=3)
        cv2.putText(img, predText, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (119, 118, 193), thickness=3)


def get_number_prediction(model, img, bottomY):
    # the input images are tensors with values in [0, 1]
    # print("input image shape...:", type(img))
    image_array = img.numpy()
    image_array = np.array(normalize(image_array), dtype=np.float32)
    img = torch.from_numpy(image_array)

    model.eval()
    with torch.no_grad():
        '''
        prediction is in the following format:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''

        prediction = model([img.to(device)])

    # print('Prediction is:', prediction)
    number, labels = get_number_from_pred(prediction, bottomY)
    return number, labels


confusing_count = 1000


def resize_img(digitPic, size):
    # put digits on standard 1000x1000 image
    ch, cw = digitPic.shape[:2]

    # find dimensions to make square
    largerDim = max(ch, cw)
    # now create large background
    background = np.zeros((largerDim, largerDim, 3), np.uint8)

    # compute xoff and yoff for placement of upper left corner of resized image
    yoff = round((largerDim-ch)/2)
    xoff = round((largerDim-cw)/2)

    background[yoff:yoff+ch, xoff:xoff+cw] = digitPic

    # lastly resize
    image = cv2.resize(background, (size, size))

    return image


def crop_confusing_digits(img, boxes, labels, scores):
    confusing_save = './ConfusingSave5and6/'
    if not os.path.isdir(confusing_save):
        os.mkdir(confusing_save)
    # any confusing digit with a score over 0.7 can be saved
    global confusing_count
    for i in range(len(boxes)):
        if labels[i] == 5 or labels[i] == 6:
            # check score
            if scores[i] >= 0.7:
                # crop digit from image
                bbox = boxes[i]
                digitCrop = img[int(bbox[1]):int(bbox[3]),
                                int(bbox[0]):int(bbox[2])]
                # resize digit crop
                digitCrop = resize_img(digitCrop, 250)
                # save digit crop
                cv2.imwrite(confusing_save +
                            str(confusing_count)+'.jpg', digitCrop)
                confusing_count += 1


def showbbox(img, bottomY):
    # the input images are tensors with values in [0, 1]
    # print("input image shape...:", type(img))
    image_array = img.numpy()
    image_array = np.array(normalize(image_array), dtype=np.float32)
    img = torch.from_numpy(image_array)

    with torch.no_grad():
        '''
        prediction is in the following format:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''

        prediction = digit_detection_model([img.to(device)])

    boxes = prediction[0]['boxes'].detach().cpu().numpy()
    labels = prediction[0]['labels'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()
    # print(prediction)
    number, boxes, labels, scores = get_number_from_pred(
        boxes, labels, scores, bottomY)

    if number is not None:
        img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
        img = (img * 255).byte().data.cpu()  # [0, 1] -> [0, 255]
        img = np.array(img)  # tensor -> ndarray

        # check if we have confusing digits
        # crop_confusing_digits(img, boxes, labels, scores)

        for i in range(len(boxes)):
            fig_draw(img, boxes[i], labels[i], scores[i])

        fig_num(img, number)

        return img, labels, scores

    return None, None, None


def get_number_from_pred_no_bottomY(boxes, labels, scores, maxDim):
    # check if length is greater than three and if so select three most distinct
    if len(labels) >= 3:
        boxes, labels, scores = no_overlap_pred(boxes, labels, scores, maxDim)
    # if only 2 or less predictions, not usable
    if len(labels) < 3:
        return None, None, None, None
    # print('boxes:', boxes)

    # get indices to sort boxes by minimum x value
    sortInd = np.argsort(boxes[:, 0])
    # put labels in order
    labels = labels[sortInd]
    boxes = boxes[sortInd]
    scores = scores[sortInd]

    # create number
    number = ''
    # iterate through each label adding it to our number
    for i in range(len(labels)):
        label = str(labels[i].item())
        # check if 10 and covert to 0
        if label == '10':
            label = '0'
            labels[i] = 0
        number += label

    return number, boxes, labels, scores

# function to check if correct or not


def updateAccuracies(pebbleActualNumber, digitAccuracy, predLabels, predScores, img):
    numberIsIncorrect = False
    scoreCode = ''
    for a in range(len(predLabels)):
        actualDigit = pebbleActualNumber[a]
        predDigit = predLabels[a]
        predScore = predScores[a]
        if predDigit == 10:
            predDigit = 0

        # check if digit is correct
        if actualDigit == predDigit:
            # now update accordingly
            if predScore < 0.8:
                digitAccuracy[1] += 1
                scoreCode += '2'
            elif predScore >= 0.8 and predScore < 0.98:
                digitAccuracy[3] += 1
                scoreCode += '4'
            else:
                digitAccuracy[5] += 1
                scoreCode += '6'
        else:
            numberIsIncorrect = True
            # now update accordingly
            if predScore < 0.8:
                digitAccuracy[0] += 1
                scoreCode += '1'
            elif predScore >= 0.8 and predScore < 0.98:
                digitAccuracy[2] += 1
                scoreCode += '3'
            else:
                digitAccuracy[4] += 1
                scoreCode += '5'

    if numberIsIncorrect:
        digitAccuracy[6] += 1
        scoreCode += '7'
    else:
        digitAccuracy[7] += 1
        scoreCode += '8'

    # put actual number in image
    scoring = str(pebbleActualNumber[0]) + str(pebbleActualNumber[1]
                                               ) + str(pebbleActualNumber[2]) + ":" + scoreCode
    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get boundary of this text
    textsize = cv2.getTextSize(scoring, font, 3, 6)[0]

    # get coords based on boundary
    textX = int((img.shape[1] - textsize[0]) / 2)
    textY = int((img.shape[0] + textsize[1]) / 2)
    cv2.putText(img, scoring, (textX, img.shape[0]-125), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 255, 255), thickness=6)

    return digitAccuracy, img


def showbox_with_accuracy(img, pebbleActualNumber, digitAccuracy):
    annImg = img.copy()
    maxDim = max(annImg.shape[0], annImg.shape[1])
    # the input images are tensors with values in [0, 1]
    # print("input image shape...:", type(img))
    transform = T.Compose([T.ToTensor()])

    img = transform(img)

    img = np.array(normalize(img.numpy()), dtype=np.float32)
    img = torch.from_numpy(img)

    with torch.no_grad():
        '''
        prediction is in the following format:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''

        prediction = digit_detection_model([img.to(device)])

    boxes = prediction[0]['boxes'].detach().cpu().numpy()
    labels = prediction[0]['labels'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()
    # print(prediction)
    number, boxes, labels, scores = get_number_from_pred_no_bottomY(
        boxes, labels, scores, maxDim)

    if number is not None:
        # img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
        # img = (img * 255).byte().data.cpu()  # [0, 1] -> [0, 255]
        # img = np.array(img)  # tensor -> ndarray

        # check if we have confusing digits
        # crop_confusing_digits(img, boxes, labels, scores)

        for i in range(len(boxes)):
            fig_draw(annImg, boxes[i], labels[i], scores[i])

        fig_num(annImg, number)

        #add in scoringe
        updateAccuracies(pebbleActualNumber, digitAccuracy, labels, scores, annImg)

        return annImg, labels, scores, digitAccuracy

    return None, None, None, digitAccuracy


def showbox_no_bottomY(img):
    annImg = img.copy()
    maxDim = max(annImg.shape[0], annImg.shape[1])
    # the input images are tensors with values in [0, 1]
    # print("input image shape...:", type(img))
    transform = T.Compose([T.ToTensor()])

    img = transform(img)

    img = np.array(normalize(img.numpy()), dtype=np.float32)
    img = torch.from_numpy(img)

    with torch.no_grad():
        '''
        prediction is in the following format:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''

        prediction = digit_detection_model([img.to(device)])

    boxes = prediction[0]['boxes'].detach().cpu().numpy()
    labels = prediction[0]['labels'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()
    # print(prediction)
    number, boxes, labels, scores = get_number_from_pred_no_bottomY(
        boxes, labels, scores, maxDim)

    if number is not None:
        # img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
        # img = (img * 255).byte().data.cpu()  # [0, 1] -> [0, 255]
        # img = np.array(img)  # tensor -> ndarray

        # check if we have confusing digits
        # crop_confusing_digits(img, boxes, labels, scores)

        for i in range(len(boxes)):
            fig_draw(annImg, boxes[i], labels[i], scores[i])

        fig_num(annImg, number)

        return annImg, labels, scores

    return None, None, None


def resize_digits_with_size_and_bottom(digitPic, size, bottomY):
    # put digits on standard 1000x1000 image
    ch, cw = digitPic.shape[:2]

    # # first resize
    # rescaleH, rescaleW = int(ch*scale), int(cw*scale)
    # digitPic = cv2.resize(digitPic, (rescaleH, rescaleW))
    # # apply to bottomY too
    # bottomY = bottomY * scale

    # find dimensions to make square
    largerDim = max(ch, cw)
    # now create large background
    background = np.zeros((largerDim, largerDim, 3), np.uint8)

    # compute xoff and yoff for placement of upper left corner of resized image
    yoff = round((largerDim-ch)/2)
    xoff = round((largerDim-cw)/2)

    background[yoff:yoff+ch, xoff:xoff+cw] = digitPic

    # add to bottomY
    bottomY = bottomY + yoff

    # lastly resize
    image = cv2.resize(background, (size, size))

    # find scale
    scale = size/largerDim
    # scale bottomY
    bottomY = bottomY * scale

    return image, bottomY


def individual_digit_detection(useCrops, imgFolder, transform, currentPebble):
    # iterate through best crops
    # print('Usable Crops:', useCrops)
    for crop, bottomY, count, c, rotation in useCrops:
        crop, bottomY = resize_digits_with_size_and_bottom(crop, 500, bottomY)
        rot_im = Image.fromarray(crop).convert("RGB")
        rot_im, _ = transform(rot_im, None)
        # predict
        # print('Usable crop count, c, rotation:', count, c, rotation)
        save_img, labels, scores = showbbox(rot_im, bottomY)
        if save_img is not None:
            # save img
            cv2.imwrite(imgFolder + "pred_" + str(count) + "_digit_crop_" +
                        str(c) + "_rot" + str(rotation)+".jpg", save_img)

            # add to pebbleDigits counts
            currentPebble.addDigits(labels, scores)
