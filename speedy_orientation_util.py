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
crop_orientation_model = torch.load(
    './saved_models/orientation_detector_complete_neg.pt')
crop_orientation_model.eval()
CLASS_NAMES = ['__background__', 'not bar', 'bar']
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
crop_orientation_model.to(device)


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


def get_coloured_mask(mask, pred):
    """
    random_colour_masks
    parameters:
        - image - predicted masks
    method:
        - the masks of each predicted object is given random colour for visualization
    """
    c = 255
    if pred == 'not bar':
        c = 0
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = [c, 0, 255]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_prediction(img, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    img = Image.fromarray(img).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img = transform(img)

    image_array = img.numpy()
    image_array = np.array(normalize(image_array), dtype=np.float32)
    img = torch.from_numpy(image_array)

    img = img.to(device)
    pred = crop_orientation_model([img])
    # print('prediction:', pred)
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > confidence]
    # take only top prediction
    if len(pred_score) == 0:
        return None, None, None, None
    if len(pred_t) == 0:
        return None, None, None, None
    masks = (pred[0]['masks'] > 0.9).detach().cpu().numpy()
    masks = np.array(masks.reshape(-1, *masks.shape[-2:]))
    # print(masks.shape)
    # print(pred[0]['labels'].numpy().max())
    pred_class = np.array([CLASS_NAMES[i]
                           for i in list(pred[0]['labels'].cpu().numpy())])
    pred_boxes = np.array([[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
                           for i in list(pred[0]['boxes'].detach().cpu().numpy())])
    masks = masks[pred_t]
    pred_boxes = pred_boxes[pred_t]
    pred_class = pred_class[pred_t]
    pred_score = np.array(pred_score)
    pred_score = pred_score[pred_t]
    return masks, pred_boxes, pred_class, pred_score


def getAngle(picCenter, barCenter, basePoint):
    # now calculate angle between three points
    ang = math.degrees(math.atan2(
        basePoint[1]-picCenter[1], basePoint[0]-picCenter[0]) - math.atan2(barCenter[1]-picCenter[1], barCenter[0]-picCenter[0]))
    return ang + 360 if ang < 0 else ang


def getCenters(boxMinP, boxMaxP, img):
    # we need to calculate the center of the box
    bx = (boxMinP[0] + boxMaxP[0])/2
    by = (boxMinP[1] + boxMaxP[1])/2

    imgH, imgW = img.shape[:2]
    ix = imgW/2
    iy = imgH/2

    barCenter = (bx, by)
    picCenter = (ix, iy)

    # calculate distance from picCenter to barCenter
    dist = math.sqrt(
        (math.pow((picCenter[0]-barCenter[0]), 2)+math.pow((picCenter[1]-barCenter[1]), 2)))

    # our third point is at the bottom of the image
    basePoint = (picCenter[0], picCenter[1]+dist)

    return barCenter, picCenter, basePoint


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


def segment_and_fix_image(img, confidence=0.9, rect_th=2, text_size=2, text_th=2):
    """
    segment_bar_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    annImg = img.copy()
    masks, boxes, pred_cls, pred_score = get_prediction(img, confidence)
    fixedImage = None
    if masks is None:
        cv2.putText(annImg, 'no detection', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 255), thickness=text_th)
    else:
        for i in range(len(pred_cls)):
            pred = pred_cls[i]
            if pred == 'bar':
                mask = masks[i]
                box = boxes[i]

                # get center of bar and image
                barCenter, picCenter, basePoint = getCenters(
                    box[0], box[1], img)

                # get angle between points
                angle = getAngle(picCenter, barCenter, basePoint)

                # get the fixed image
                fixedImage = rotate_im(annImg.copy(), -1*angle)
                cv2.putText(annImg, 'Angle:' + str(round(angle, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 0), thickness=text_th)
                # print('Horizontal not found')
                cv2.line(annImg, tuple(map(round, barCenter)), tuple(
                    map(round, picCenter)), (255, 255, 255), 10)
                cv2.line(annImg, tuple(map(round, picCenter)), tuple(
                    map(round, basePoint)), (255, 255, 255), 10)
                rgb_mask = get_coloured_mask(mask, pred)
                annImg = cv2.addWeighted(annImg, 1, rgb_mask, 0.5, 0)
                cv2.rectangle(annImg, box[0], box[1],
                              color=(0, 255, 0), thickness=rect_th)
                cv2.putText(annImg, pred+": "+str(pred_score[i]), box[0], cv2.FONT_HERSHEY_SIMPLEX,
                            text_size, (0, 255, 0), thickness=text_th)

                break

    return annImg, fixedImage


def segment_and_fix_frame_range(frame, confidence=0.9, rect_th=2, text_size=2, text_th=2):
    """
    segment_bar_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    annImg = frame.copy()
    masks, boxes, pred_cls, pred_score = get_prediction(frame, confidence)
    fixedImages = []
    errorRange = [-5, 0, 5]
    if masks is None:
        cv2.putText(annImg, 'no detection', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 255), thickness=text_th)
    else:
        for i in range(len(pred_cls)):
            pred = pred_cls[i]
            if pred == 'bar':
                mask = masks[i]
                box = boxes[i]

                # get center of bar and image
                barCenter, picCenter, basePoint = getCenters(
                    box[0], box[1], frame)

                # get angle between points
                angle = getAngle(picCenter, barCenter, basePoint)

                # get the fixed image
                for errorVal in errorRange:
                    fixedImages.append(
                        rotate_im(frame.copy(), -1*(angle + errorVal)))
                cv2.putText(annImg, 'Angle:' + str(round(angle, 2)), (50, 250), cv2.FONT_HERSHEY_SIMPLEX,
                            10, (255, 255, 0), thickness=10)
                # print('Horizontal not found')
                cv2.line(annImg, tuple(map(round, barCenter)), tuple(
                    map(round, picCenter)), (255, 255, 255), 30)
                cv2.line(annImg, tuple(map(round, picCenter)), tuple(
                    map(round, basePoint)), (255, 255, 255), 30)
                rgb_mask = get_coloured_mask(mask, pred)
                annImg = cv2.addWeighted(annImg, 1, rgb_mask, 0.5, 0)
                cv2.rectangle(annImg, box[0], box[1],
                              color=(0, 255, 0), thickness=30)
                cv2.putText(annImg, pred+": "+str(round(pred_score[i], 4)), (box[0][0], box[0][1]-30), cv2.FONT_HERSHEY_SIMPLEX,
                            7, (0, 255, 0), thickness=7)

                break

    return annImg, fixedImages


def segment_and_fix_image_range(img, og_img, confidence=0.9, rect_th=2, text_size=2, text_th=2):
    """
    segment_bar_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    annImg = img.copy()
    masks, boxes, pred_cls, pred_score = get_prediction(img, confidence)
    fixedImages = []
    errorRange = [-5, 0, 5]
    if masks is None:
        cv2.putText(annImg, 'no detection', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 255), thickness=text_th)
    else:
        for i in range(len(pred_cls)):
            pred = pred_cls[i]
            if pred == 'bar':
                mask = masks[i]
                box = boxes[i]

                # get center of bar and image
                barCenter, picCenter, basePoint = getCenters(
                    box[0], box[1], img)

                # get angle between points
                angle = getAngle(picCenter, barCenter, basePoint)

                # get the fixed image
                for errorVal in errorRange:
                    fixedImages.append(
                        rotate_im(og_img.copy(), -1*(angle + errorVal)))
                cv2.putText(annImg, 'Angle:' + str(round(angle, 2)), (50, 75), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (255, 255, 0), thickness=3)
                # print('Horizontal not found')
                cv2.line(annImg, tuple(map(round, barCenter)), tuple(
                    map(round, picCenter)), (255, 255, 255), 10)
                cv2.line(annImg, tuple(map(round, picCenter)), tuple(
                    map(round, basePoint)), (255, 255, 255), 10)
                rgb_mask = get_coloured_mask(mask, pred)
                annImg = cv2.addWeighted(annImg, 1, rgb_mask, 0.5, 0)
                cv2.rectangle(annImg, box[0], box[1],
                              color=(0, 255, 0), thickness=3)
                cv2.putText(annImg, pred+": "+str(round(pred_score[i], 4)), (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                            text_size, (0, 255, 0), thickness=text_th)

                break

    return annImg, fixedImages


def rotateWithSlightError(img):
    errorRange = [-5, 0, 5]
    fixedImages = []
    # get the fixed image
    for errorVal in errorRange:
        fixedImages.append(rotate_im(img.copy(), errorVal))

    return fixedImages
