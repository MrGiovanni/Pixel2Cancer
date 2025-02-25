from collections import defaultdict
import numpy as np
from random import choice
from cellular.cellular import update_cellular
import torch
import argparse
import SimpleITK as sitk
from skimage.measure import label, regionprops
import cv2
import math
from .utils_visual import generate_tumor,get_predefined_texture

Organ_List = {'liver': [1,2], 'pancreas': [1,2], 'kidney': [1,2]}
Organ_HU = {'liver': [100, 160],'pancreas': [100, 160], 'kidney': [140, 200]}

def main():
    steps = 50  # step
    organ_name = 'liver'  # organ name
    start_point = (0, 0, 0)  # start point
    img = sitk.ReadImage('/home/ylai/code/Tumor_Growth/data/img/FELIX-CYS-1000_VENOUS.nii.gz')
    label = sitk.ReadImage('/home/ylai/code/Tumor_Growth/data/mask/FELIX-CYS-1000_VENOUS.nii.gz') # organ_mask
    save_img_path = 'test_tumor.nii.gz'
    save_label_path = 'test_label_tumor.nii.gz'
    
    organ_hu_lowerbound = Organ_HU[organ_name][0]  # organ hu lowerbound
    outrange_standard_val = Organ_HU[organ_name][1]  # outrange standard value
    organ_standard_val = 0  # organ standard value
    threshold = 10  # threshold
    kernel_size = (3, 3, 3)  # Receptive Field
    
    textures = []
    sigma_as = [3, 6, 9, 12, 15]
    sigma_bs = [4, 7]
    predefined_texture_shape = (500, 500, 500)
    for sigma_a in sigma_as:
        for sigma_b in sigma_bs:
            texture = get_predefined_texture(predefined_texture_shape, sigma_a, sigma_b)
            textures.append(texture)
    print("All predefined texture have generated.")
    
    img = sitk.GetArrayFromImage(img)
    label = sitk.GetArrayFromImage(label)
    
    
    img,label = generate_tumor(img,label, texture, steps, kernel_size, organ_standard_val, organ_hu_lowerbound, outrange_standard_val, threshold, organ_name, start_point)
    
    img = sitk.GetImageFromArray(img)
    label = sitk.GetImageFromArray(label)
    sitk.WriteImage(img, save_img_path)
    sitk.WriteImage(label, save_label_path) 
    
if __name__ == "__main__":
    main()
