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
from utils import generate_tumor

Organ_List = {'liver': [6,15]}
Organ_HU = {'liver': [100, 160]}

def main():
    steps = 50  # step
    kernel_size = (3, 3, 3)  # Receptive Field
    organ_hu_lowerbound = Organ_HU['liver'][0]  # organ hu lowerbound
    outrange_standard_val = Organ_HU['liver'][1]  # outrange standard value
    organ_standard_val = 0  # organ standard value
    threshold = 10  # threshold
    organ_name = 'liver'  # organ name
    img = sitk.ReadImage('/home/ylai/code/Tumor_Growth/data/img/FELIX-CYS-1000_VENOUS.nii.gz')
    img = sitk.GetArrayFromImage(img)
    label = sitk.ReadImage('/home/ylai/code/Tumor_Growth/data/mask/FELIX-CYS-1000_VENOUS.nii.gz')
    label = sitk.GetArrayFromImage(label)
    
    img,label = generate_tumor(img,label, steps, kernel_size, organ_standard_val, organ_hu_lowerbound, outrange_standard_val, threshold, organ_name)
    
    img = sitk.GetImageFromArray(img)
    label = sitk.GetImageFromArray(label)
    sitk.WriteImage(img, 'test_tumor.nii.gz')
    sitk.WriteImage(label, 'test_label_tumor.nii.gz') 
if __name__ == "__main__":
    main()