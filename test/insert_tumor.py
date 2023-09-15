# Load CT images

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import SimpleITK as sitk

# Load the image
img_path = './data/img/FELIX-CYS-1000_VENOUS.nii.gz'
img = sitk.ReadImage(img_path)
img = sitk.GetArrayFromImage(img)
print(img.shape)
# Load the mask
mask_path = './data/mask/FELIX-CYS-1000_VENOUS.nii.gz'
mask = sitk.ReadImage(mask_path)
mask = sitk.GetArrayFromImage(mask)

#load the tumor
tumor_path = './data/tumor/tumor1.npy'
tumor = np.load(tumor_path)[10]
tumor_shape = tumor.shape


#map numpy tumor to ct value
condition = np.where(tumor == 1)
num = np.sum(tumor)
random_value = np.random.randint(70,90,num)
tumor[condition] = random_value

#insert the tumor into the image
# insert_index = np.where(mask == 6)
insert_index = (378,162,150)
print(insert_index)
img[insert_index[0]:insert_index[0]+ tumor_shape[0], insert_index[1]:insert_index[1]+tumor_shape[1], insert_index[2]: insert_index[2]+tumor_shape[2]] -= tumor

#save the image
img = sitk.GetImageFromArray(img)
sitk.WriteImage(img, './data/new/FELIX-CYS-1000_VENOUS_tumor.nii.gz')


