import sys
from xml.dom.minidom import parse
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import torch
import numpy as np
import xml
import math
import shutil
import xml.etree.ElementTree as ET
import torchvision.transforms as VT
import train_utils.transforms as T

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')
# utils transforms, engine are the utils.py, transforms.py, engine.py under this fold

# %matplotlib inline


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
digit_segmentation_model = torch.load(
    r'./saved_models/group-seg-multi-100.pt')
digit_segmentation_model.to(device)
digit_segmentation_model.eval()


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


def digit_segmentation(img):
    # the input images are tensors with values in [0, 1]
    # print("input image shape...:", type(img))
    h, w = img.shape[:2]
    transform = VT.Compose([VT.ToTensor()])
    img = transform(img)

    image_array = img.numpy()
    image_array = np.array(normalize(image_array), dtype=np.float32)
    img = torch.from_numpy(image_array)

    with torch.no_grad():
        '''
        prediction is in the following format:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]a
        '''

        prediction = digit_segmentation_model([img.to(device)])

    # print(prediction)

    img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
    img = (img * 255).byte().data.cpu()  # [0, 1] -> [0, 255]
    img = np.array(img)  # tensor -> ndarray

    bboxes = prediction[0]['boxes'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()
    predictions = prediction[0]['labels'].detach().cpu().numpy()

    masks = (prediction[0]['masks'] > 0.5).detach().cpu().numpy()
    masks = np.array(masks.reshape(-1, *masks.shape[-2:]))

    # print(bboxes)
    goodBBoxes = []
    goodScores = []
    goodPredictions = []
    goodMasks = []
    # create digit crops
    digitCrops = []
    originalDigitCrops = []
    for i in range(len(scores)):
        if scores[i] >= 0.99:
            if predictions[i] == 2:
                bbox = bboxes[i].astype(int)
                goodBBoxes.append(bbox)
                goodScores.append(scores[i])
                goodPredictions.append(predictions[i])
                goodMasks.append(masks[i])
                digits_crop = img[max(0, bbox[1]-100):min(h, bbox[3]+100),
                                  max(0, bbox[0]-100):min(w, bbox[2]+100)]
                original_digit_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                digitCrops.append(digits_crop)
                originalDigitCrops.append(original_digit_crop)
        else:
            # scores already sorted
            break

    # draw boxes if they exist
    if (len(goodBBoxes) != 0):
        return digitCrops, goodBBoxes, goodScores, goodPredictions, goodMasks, originalDigitCrops
    return None, None, None, None, None, None


def get_coloured_mask(mask):
    """
    random_colour_masks
    parameters:
        - image - predicted masks
    method:
        - the masks of each predicted object is given random colour for visualization
    """
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = [255, 0, 255]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def draw_bboxes(img, bboxes, scores, predictions, masks):
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin = round(bbox[0])
        ymin = round(bbox[1])
        xmax = round(bbox[2])
        ymax = round(bbox[3])
        if predictions[i] == 1:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          (0, 0, 255), thickness=1)
            cv2.putText(img, str(predictions[i])+':'+str(scores[i]), (xmin, ymin+25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), thickness=2)
        else:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          (0, 255, 0), thickness=1)
            cv2.putText(img, 'digits:'+str(scores[i]), (xmin, ymin+25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), thickness=2)

            rgb_mask = get_coloured_mask(masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    return img