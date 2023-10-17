import os
import yaml
import cv2
from algorithms import find_inter_means_threshold,find_inter_means_threshold_scikit,segment_image

def load_config(file_path = "./config.yaml"):
    with open(file_path,"r") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    return config

def process_image(image,filename,mode='manual',plot="True"):
    if mode=='auto':
        threshold = find_inter_means_threshold_scikit(image)
    else:
        threshold = find_inter_means_threshold(image,filename,plot)
    segmented_image = segment_image(image, threshold)
    return segmented_image

def write_image(image, filename, output_folder):
    output_filename = os.path.splitext(filename)[0] + '_segmented.jpg'
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, image)
    return image
