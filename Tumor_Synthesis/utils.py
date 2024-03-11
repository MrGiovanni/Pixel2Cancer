import numpy as np
from random import choice
from .cellular import update_cellular
import torch
from skimage.measure import label, regionprops
import cv2
import math
import os
from scipy.ndimage import gaussian_filter
import random

Organ_List = {'liver': [1,2], 'pancreas': [1,2], 'kidney': [1,2]}
Organ_HU = {'liver': [100, 160],'pancreas': [100, 160], 'kidney': [140, 200]}
Tumor_size = {'liver': 50, 'pancreas': 30, 'kidney': 30}

def get_predefined_texture(mask_shape, sigma_a, sigma_b):
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    a_2 = gaussian_filter(a, sigma=sigma_a)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a =  scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12    # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta*b, 0, 1) # 目前是0-1区间

    return Bj

def grow_tumor(current_state, density_organ_state, kernel_size, steps, all_states, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map, core_point, organ_name,args):
    # process

    death_flag = False
    # simulate tumor growth for each step
    for i in range(steps+1):
        current_state = update_cellular(current_state, density_organ_state, (kernel_size[0], kernel_size[1], kernel_size[2]), (
            organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold), death_flag).clamp(max=(outrange_standard_val + 2))
    
    #When tumor larger than 2cm process Death: if tumor surpasses the threshold, it may die.
    death = np.random.randint(0, 2) 

    if death == 0:
        if steps > Tumor_size[organ_name]:
            death_flag = True
            current_state = current_state.cpu().numpy().copy()
            current_state[current_state >= outrange_standard_val] = 0
            current_state[current_state >= threshold] = threshold
            core_point = np.array(core_point)
            num_death_point = np.random.randint(1, core_point.shape[0]+1)
            
            random_index = np.random.choice(core_point.shape[0],size=core_point.shape[0], replace=False)
            death_point = core_point[random_index]
            current_state[death_point[:, 0], death_point[:, 1], death_point[:, 2]] = -1
            for point in death_point:
                idx_x, idx_y, idx_z = point
                num_expend = np.random.randint(5,20)
                
                bias = steps/10 - 2
                for i in range(num_expend):
                    x_offset = np.random.randint(-bias,bias)
                    y_offset = np.random.randint(-bias,bias)
                    z_offset = np.random.randint(-bias,bias)
                    
                    try:
                        if current_state[idx_x + x_offset, idx_y + y_offset, idx_z + z_offset] == threshold:
                            current_state[idx_x + x_offset, idx_y + y_offset, idx_z + z_offset] = -1
                    except:
                        pass
            
            # death spread

            current_state = torch.tensor(current_state, dtype=torch.int32).cuda(args.gpu)
            for i in range(int(steps/7 + 1)):
                current_state = update_cellular(current_state, density_organ_state, (kernel_size[0], kernel_size[1], kernel_size[2]), (
                    organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold), death_flag).clamp(max=(outrange_standard_val + 2))

        


    

    state = current_state.cpu().numpy().copy()
    state = np.array(state)
   
    
    
    # postprocess
    if death_flag == False:
        state[state >= outrange_standard_val] = 0
        state[state >= threshold] = threshold

    # Blur the tumor map
    temp = state.astype(np.int16)
    # kernel = (3, 3)

    # for z in range(temp.shape[0]):
    #     temp[z] = cv2.GaussianBlur(temp[z], kernel, 0)
    #     # temp[z] = cv2.filter2D(temp[z], -1, sharpen_kernel)

    # save the tumor map
    state = temp

    return state, death_flag

    # for i in range(steps):
    #     cpl3d.plot3d(all_states, timestep=i)


def mass_effect(img, tumor, start_point):
    '''
    Mass effect transformation of a image region
    '''
    # preprocessing
    tumor[(tumor > 0)] = 1
    x = start_point[1]
    y = start_point[2]
    for c in range(img.shape[0]):
        if tumor[c, x, y] == 1:
            radius = 0
            while tumor[c, x + radius, y] == 1:
                radius += 1
    
            img[c] = expand(img[c], y, x, 1.3 * radius, 30) 
                        
                
    return img


def expand(src, PointX, PointY, Radius, Strength):
    '''
    Expansion transformation of a image region
    https://www.jb51.net/article/230354.htm
    Param:
        src (2d ndarray): source image
        PointX (int): center in the X axis
        PointY (int): center in the Y axis
        Radius (int): the radius of expanded area
        Strength (int): the expansion strength, 0-100
    Return:
        processed_image (2d ndarray): expanded image
    '''
    processed_image = np.zeros(src.shape, np.uint8)
    processed_image = src.copy()
    height = src.shape[0]
    width = src.shape[1]
    PowRadius = Radius * Radius

    maskImg = np.zeros(src.shape[:2], np.uint8)
    cv2.circle(maskImg, (PointX, PointY),
               math.ceil(Radius), (255, 255, 255), -1)

    mapX = np.vstack([np.arange(width).astype(
        np.float32).reshape(1, -1)] * height)
    mapY = np.hstack([np.arange(height).astype(
        np.float32).reshape(-1, 1)] * width)

    OffsetX = mapX - PointX
    OffsetY = mapY - PointY
    XY = OffsetX * OffsetX + OffsetY * OffsetY

    ScaleFactor = 1 - XY / PowRadius
    ScaleFactor = 1 - Strength / 100 * ScaleFactor
    UX = OffsetX * ScaleFactor + PointX
    UY = OffsetY * ScaleFactor + PointY
    UX[UX < 0] = 0
    UX[UX >= width] = width - 1
    UY[UY < 0] = 0
    UY[UY >= height] = height - 1

    np.copyto(UX, mapX, where=maskImg == 0)
    np.copyto(UY, mapY, where=maskImg == 0)

    UX = UX.astype(np.float32)
    UY = UY.astype(np.float32)

    processed_image = cv2.remap(src, UX, UY, interpolation=cv2.INTER_LINEAR)

    return processed_image

def Quantify(processed_organ_region, organ_hu_lowerbound, organ_standard_val, outrange_standard_val):
    
    # Quantify the intensity of differnent part of the organ
    interval = (outrange_standard_val - organ_hu_lowerbound) / 3
    processed_organ_region[(processed_organ_region < (
        organ_hu_lowerbound+interval))] = organ_hu_lowerbound
    processed_organ_region[(processed_organ_region >= (organ_hu_lowerbound+interval)) & (
        processed_organ_region < (organ_hu_lowerbound + 2*interval))] = organ_hu_lowerbound + interval
    processed_organ_region[(processed_organ_region >= (organ_hu_lowerbound + 2*interval)) & (
        processed_organ_region < outrange_standard_val)] = organ_hu_lowerbound + 2*interval
    

    density_organ_map = processed_organ_region.copy()

    processed_organ_region[processed_organ_region <
                           outrange_standard_val] = organ_standard_val
    processed_organ_region[processed_organ_region == outrange_standard_val] = 1
    

    
    processed_organ_region[processed_organ_region == 1] = outrange_standard_val

    density_organ_map[(density_organ_map == outrange_standard_val) & (
        processed_organ_region != outrange_standard_val)] = organ_hu_lowerbound + 2*interval
    
  
    
    return processed_organ_region, density_organ_map


def map_to_CT_value(img, tumor, texture, density_organ_map, threshold, outrange_standard_val, organ_hu_lowerbound, organ_standard_val, start_point, death_flag, organ):

    img = img.astype(np.float32)
    tumor = tumor.astype(np.float32)
    density_organ_map = density_organ_map.astype(np.float32)

    tumor_1 = tumor.copy()
    tumor_1[tumor_1 == -1] = threshold
    kernel = (3, 3)

    for z in range(tumor_1.shape[0]):
        tumor_1[z] = cv2.GaussianBlur(tumor_1[z], kernel, 0)

    # deal with the conflict vessel
    interval = (outrange_standard_val - organ_hu_lowerbound) / 3
    if organ == 'liver' or organ == 'kidney':
        # deal with the conflict vessel
        vessel_condition = (density_organ_map == outrange_standard_val) & (tumor_1 >= threshold/2)

        # deal with the high intensity tissue
        high_tissue_condition = (density_organ_map == (organ_hu_lowerbound + 2 * interval)) & (tumor_1 != 0)

        img[vessel_condition] *= (organ_hu_lowerbound + interval/2) / outrange_standard_val
        img[high_tissue_condition] *= (organ_hu_lowerbound + 2 * interval) / outrange_standard_val
        # print('end change CT value')
    elif organ == 'pancreas':
        
        # deal with the conflict vessel
        vessel_condition = (density_organ_map == outrange_standard_val) & (tumor_1 >= threshold/2)

        # deal with the high intensity tissue
        high_tissue_condition = (density_organ_map == (organ_hu_lowerbound + 2 * interval)) & (tumor_1 != 0)

        img[vessel_condition] *= (organ_hu_lowerbound + interval/2) / outrange_standard_val
        img[high_tissue_condition] *= (organ_hu_lowerbound + 2 * interval) / outrange_standard_val
        
        fat_condition = (img <= 65) & (tumor_1 != 0)
        tumor_region_mean = np.mean(img[tumor_1 != 0])
        differnece = np.random.uniform(tumor_region_mean, 2*tumor_region_mean)

        tumor_fat = tumor_1.copy()
        tumor_fat[~fat_condition] = 0
        img = img + texture*differnece*tumor_fat/threshold
        

    # typically difference between tumor and normal tissue
    difference_1 = np.random.uniform(65, 145)

    map_img = img - texture*difference_1*tumor_1/threshold
    
    # death cell mapping
    if death_flag == True:
        tumor_2 = tumor.copy()
        tumor_2[tumor_2 > -1] = 0
        kernel = (3, 3)
        for z in range(tumor_2.shape[0]):
            tumor_2[z] = cv2.GaussianBlur(tumor_2[z], kernel, 0)

        difference_2 = np.random.uniform(90, 110)
        map_img = map_img + texture*difference_2*tumor_2
        

    # # postprocess
    tumor_region = map_img.copy()
    tumor_region[tumor_1 == 0] = 0

    for z in range(tumor_region.shape[0]):
        tumor_region[z] = cv2.GaussianBlur(tumor_region[z], kernel, 0)
    if organ == 'liver':
        map_img[(tumor_1 >= threshold/2) & (density_organ_map >= (organ_hu_lowerbound + 2 * interval))] = tumor_region[(tumor_1 >= threshold/2) & (density_organ_map >= (organ_hu_lowerbound + 2 * interval))]
    elif organ == 'pancreas':
        map_img[(tumor_1 >= threshold/2) & (density_organ_map >= (organ_hu_lowerbound + 2 * interval))] = tumor_region[(tumor_1 >= threshold/2) & (density_organ_map >= (organ_hu_lowerbound + 2 * interval))]
        map_img[(tumor_1 >= threshold/2) & fat_condition] = tumor_region[(tumor_1 >= threshold/2) & fat_condition]
        map_img[(tumor_1 >= threshold/2) & (map_img < 0)] = tumor_region[(tumor_1 >= threshold/2) & (map_img < 0)]
    
    map_img = map_img.astype(np.int16)
    tumor   = tumor.astype(np.int16)
    

    return map_img



def generate_tumor(img, mask, texture,steps, kernel_size, organ_standard_val, organ_hu_lowerbound, outrange_standard_val, threshold, organ_name, args):
    
    steps = np.random.randint(5, steps)
    
    # load organ and quantify
    # get the organ region
    organ_region = np.where(np.isin(mask, Organ_List[organ_name]))
    min_x, max_x = np.min(organ_region[0]), np.max(organ_region[0])
    min_y, max_y = np.min(organ_region[1]), np.max(organ_region[1])
    min_z, max_z = np.min(organ_region[2]), np.max(organ_region[2])


    # random select a start point
    
    # crop the organ region
    cropped_organ_region = mask[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    cropped_img = img[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1].copy()

    x_length, y_length, z_length = max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1

    start_x = random.randint(0, texture.shape[0] - x_length - 1) # random select the start point, -1 is to avoid boundary check
    start_y = random.randint(0, texture.shape[1] - y_length - 1) 
    start_z = random.randint(0, texture.shape[2] - z_length - 1) 
    cropped_texture = texture[start_x:start_x+x_length, start_y:start_y+y_length, start_z:start_z+z_length]

    
    # Quantify the density of the organ
    select_organ_region = np.isin(cropped_organ_region, Organ_List[organ_name])
    processed_organ_region = cropped_img.copy()
    processed_organ_region[~select_organ_region] = outrange_standard_val
    processed_organ_region[processed_organ_region >
                        outrange_standard_val] = outrange_standard_val


    # Quantify the density of the organ
    processed_organ_region, density_organ_map = Quantify(processed_organ_region, organ_hu_lowerbound, organ_standard_val, outrange_standard_val)

    # initialize state maps
    current_state = torch.tensor(
        processed_organ_region, dtype=torch.int32).cuda(args.gpu)
    density_organ_state = torch.tensor(
        density_organ_map, dtype=torch.int32).cuda(args.gpu)
    # sample the initial tumor number
    try_time = 0
    try_max = np.random.randint(1, 10)
    core_point = []
    while try_time < try_max:
        try_time += 1
        # select point in organ region
        matching_indices = np.argwhere(
            processed_organ_region == organ_standard_val)
        if matching_indices.size > 0:
            random_index = np.random.choice(matching_indices.shape[0])
            start_point = matching_indices[random_index]
            large = np.random.randint(0,3)
            idx_x, idx_y, idx_z = start_point
            core_point.append(start_point)
            processed_organ_region[idx_x, idx_y, idx_z] = threshold/2  # start point initialize
            current_state[idx_x, idx_y, idx_z] = threshold/2
            if large == 0:
                num_expend = np.random.randint(3,10)

                for i in range(num_expend):
                    x_offset = np.random.randint(-10,10)
                    y_offset = np.random.randint(-10,10)
                    z_offset = np.random.randint(-10,10)
                    try:
                        if processed_organ_region[idx_x + x_offset, idx_y + y_offset, idx_z + z_offset] == organ_standard_val:
                            processed_organ_region[idx_x + x_offset, idx_y + y_offset, idx_z + z_offset] = threshold/2
                            current_state[idx_x + x_offset, idx_y + y_offset, idx_z + z_offset] = threshold/2
                            core_point.append([idx_x + x_offset, idx_y + y_offset, idx_z + z_offset])
                    except:
                        pass
                        
        
        
        current_state[idx_x, idx_y, idx_z] = threshold/2  # start point initialize

    all_states = []  # states of each step

    # simulate tumor growth
    tumor_out,death_flag = grow_tumor(current_state, density_organ_state, kernel_size, steps, all_states,
                        organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map, core_point, organ_name, args)

    # map to CT value

    tumor_out[cropped_organ_region==0] = 0

    img_out = map_to_CT_value(cropped_img, tumor_out, cropped_texture, density_organ_map, threshold, outrange_standard_val, organ_hu_lowerbound, organ_standard_val, start_point, death_flag, organ_name)

    # save the result
    img[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = img_out

    mask_tumor = np.zeros_like(mask)
    mask_tumor[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = tumor_out
    mask_tumor[mask_tumor > 0] = 1
    mask_tumor[mask==0] = 0
    mask_out = mask + mask_tumor
    
    mask_out[mask_out > 2] = 2

    return img, mask_out

    

    

    

    
    

 


    
