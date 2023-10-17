import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_isodata # used for auto mode only
import os

def find_inter_means_threshold(image,filename,plot = True):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = np.mean(gray_image)

    thresholds = [threshold]  
    while True:
        region1 = gray_image[gray_image <= threshold]
        region2 = gray_image[gray_image > threshold]

        mean1 = np.mean(region1) if len(region1) > 0 else 0
        mean2 = np.mean(region2) if len(region2) > 0 else 0

        new_threshold = (mean1 + mean2) / 2

        thresholds.append(new_threshold)  
        if abs(threshold - new_threshold) < 1e-2:
            break

        threshold = new_threshold

    threshold = int(threshold)

    if plot=="True":
        directory = 'histograms'
        os.makedirs(directory, exist_ok=True)
        colors = [
        'lightblue',
        'lightgreen',
        'lightpink',
        'lightseagreen',
        'lightcoral',
        'lightyellow',
        'lightcyan',
        'lightsteelblue',
        'lightsalmon',
        'lightgray'
        ]

        plt.figure(figsize=(50, 10))
        plt.subplot(1, 2, 1)
        plt.hist(gray_image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
        for i, t in enumerate(thresholds[:-1]):
            color = colors[i % len(colors)]  
            plt.axvline(x=t, color=color, linestyle='--', linewidth=1)
        plt.axvline(x=thresholds[-1], color='r', linestyle='--', linewidth=3, label=f'Final Threshold {round(thresholds[-1],3)}')
        plt.legend()
        plt.title(f'{filename} : Histogram with Thresholds',fontsize=30)

        plt.xticks(range(0, 256, 10))
        plt.savefig(f'{directory}/{filename.split(".")[0]}_histogram.jpg', bbox_inches='tight')

    return threshold

def find_inter_means_threshold_scikit(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_value = threshold_isodata(image)
    return threshold_value

def segment_image(image, threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segmented_image = (gray_image > threshold).astype(np.uint8) * 255
    return segmented_image