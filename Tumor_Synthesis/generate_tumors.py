import numpy as np
from random import choice
from cellular import update_cellular
import torch
import argparse
import SimpleITK as sitk
from skimage.measure import label, regionprops
import cv2
import math
# import datafolds.datafold_read as datafold_read
import os

Organ_List = {'liver': [1,2], 'liver2': [6, 15], 'pancreas': [11]}
Organ_HU = {'liver': [100, 160]}


def grow_tumor(current_state, density_organ_state, kernel_size, steps, all_states, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map):
    # process
    original_state = current_state.cpu().numpy().copy()
    for i in range(steps+1):
        current_state = update_cellular(current_state, density_organ_state, (kernel_size[0], kernel_size[1], kernel_size[2]), (
            organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold)).clamp(max=(outrange_standard_val + 2))
        temp = current_state.cpu().numpy().copy()
        # print(np.sum(temp==0))
        all_states.append(temp)

    all_states = np.array(all_states)

    # postprocess
    all_states[all_states >= outrange_standard_val] = 0
    all_states[all_states >= threshold] = threshold

    # Blur the tumor map
    temp = all_states[steps].astype(np.int16)
    kernel = (3, 3)

    for z in range(temp.shape[0]):
        temp[z] = cv2.GaussianBlur(temp[z], kernel, 0)
        # temp[z] = cv2.filter2D(temp[z], -1, sharpen_kernel)

    # save the tumor map
    all_states[steps] = temp

    unique_values, counts = np.unique(all_states[steps], return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"value: {value} times: {count} ")
    return all_states

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
            print(c, radius)
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
    # Quantify the density of the organ
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
    binary_array = processed_organ_region.copy()

    # size
    k = 20000
    # Unicom domain
    labeled_array, num_features = label(
        binary_array, connectivity=3, return_num=True)
    regions = regionprops(labeled_array)
    new_array = np.zeros_like(binary_array)

    # add the region
    for region in regions:
        if region.area > k:
            label_value = region.label
            new_array[labeled_array == label_value] = 1

    processed_organ_region = new_array.copy()

    print(processed_organ_region.shape)
    # save
    

    processed_organ_region[processed_organ_region == 1] = outrange_standard_val

    density_organ_map[(density_organ_map == outrange_standard_val) & (
        processed_organ_region != outrange_standard_val)] = organ_hu_lowerbound + 2*interval
    
  
    
    return processed_organ_region, density_organ_map

def map_to_CT_value(img, tumor, density_organ_map, steps, threshold, outrange_standard_val, organ_hu_lowerbound, organ_standard_val, start_point):

    img = img.astype(np.float32)
    tumor = tumor[steps].astype(np.float32)
    density_organ_map = density_organ_map.astype(np.float32)



    # deal with the conflict vessel
    interval = (outrange_standard_val - organ_hu_lowerbound) / 3
    vessel_condition = (density_organ_map == outrange_standard_val) & (tumor >= threshold/2)
    vessel_value = np.random.randint(40, 50, tumor.shape, dtype=np.int16)

    # deal with the high intensity tissue
    high_tissue_condition = (density_organ_map == (organ_hu_lowerbound + 2 * interval)) & (tumor != 0)
    high_tissue_value = np.random.randint(20, 30, tumor.shape, dtype=np.int16)

    kernel = (3, 3)
    for z in range(vessel_value.shape[0]):
        vessel_value[z] = cv2.GaussianBlur(vessel_value[z], kernel, 0)
        high_tissue_value[z] = cv2.GaussianBlur(high_tissue_value[z], kernel, 0)

    img[vessel_condition] *= (organ_hu_lowerbound + interval/2) / outrange_standard_val
    img[high_tissue_condition] *= (organ_hu_lowerbound + 2 * interval) / outrange_standard_val

    

    # random tumor value
    tumor_value = np.random.randint(5, 15, tumor.shape, dtype=np.int16)
    tumor_value[tumor == 0] = 0

    # blur the tumor value
    kernel = (3, 3)
    for z in range(tumor_value.shape[0]):
        tumor_value[z] = cv2.GaussianBlur(tumor_value[z], kernel, 0)

    # CT mapping function
    # map_img = img * -(tumor/40 - 1) + tumor/50 * tumor_value

    bias = np.random.uniform(1,4)

    map_img = img * - (tumor / (threshold + bias * threshold) - 1) - (1 - tumor/(threshold + 0)) * tumor_value

    # postprocess
    tumor_region = map_img.copy()
    tumor_region[tumor == 0] = 0
    for z in range(tumor_region.shape[0]):
        tumor_region[z] = cv2.GaussianBlur(tumor_region[z], kernel, 0)
    map_img[(tumor >= threshold/2) & (density_organ_map >= (organ_hu_lowerbound + 2 * interval))] = tumor_region[(tumor >= threshold/2) & (density_organ_map >= (organ_hu_lowerbound + 2 * interval))]

    map_img = map_img.astype(np.int16)

    

    return map_img



def main():

    # train_list, val_list = datafold_read.datafold_read(
    #     './datafolds/healthy.json', '/ccvl/net/ccvl15/zzhou82/PublicAbdominalData')
    # print(train_list)

    file_list = '../list/healthy_list.csv'
    with open(file_list, 'r') as f:
        lines = f.readlines()
    train_list = [line.strip() for line in lines]
    print(train_list)
    save_path = '/ccvl/net/ccvl15/yuxiang/04_LITS'
     # initial
    steps = 50  # step
    kernel_size = (3, 3, 3)  # Receptive Field
    organ_hu_lowerbound = Organ_HU['liver'][0]  # organ hu lowerbound
    outrange_standard_val = Organ_HU['liver'][1]  # outrange standard value
    organ_standard_val = 0  # organ standard value
    threshold = 10  # threshold
    
    save_list = []

    for file in train_list:
       

        img = sitk.ReadImage(file + '/ct.nii.gz')
        img = sitk.GetArrayFromImage(img)
        mask = sitk.ReadImage(file + '/original_label.nii.gz')
        mask = sitk.GetArrayFromImage(mask)

        # load organ and quantify
        # get the organ region
        organ_region = np.where(np.isin(mask, Organ_List['liver']))
        min_x = min(organ_region[0])
        max_x = max(organ_region[0])
        min_y = min(organ_region[1])
        max_y = max(organ_region[1])
        min_z = min(organ_region[2])
        max_z = max(organ_region[2])


        # random select a start point
        for i in range(10):
             # crop the organ
            cropped_organ_region = mask[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1].copy()
            cropped_img = img[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1].copy()
            print(cropped_organ_region.shape)
            print(cropped_img.shape)

            # Quantify the density of the organ
            select_organ_region = np.isin(cropped_organ_region, Organ_List['liver'])
            processed_organ_region = cropped_img.copy()
            processed_organ_region[~select_organ_region] = outrange_standard_val
            processed_organ_region[processed_organ_region >
                                outrange_standard_val] = outrange_standard_val

        

            processed_organ_region, density_organ_map = Quantify(processed_organ_region, organ_hu_lowerbound, organ_standard_val, outrange_standard_val)
            save = sitk.GetImageFromArray(density_organ_map)  
            sitk.WriteImage(save, '../test_result/test_soft.nii.gz')


            current_state = torch.tensor(
                processed_organ_region, dtype=torch.int32).cuda(device='cuda:0')
            density_organ_state = torch.tensor(
                density_organ_map, dtype=torch.int32).cuda(device='cuda:0')
            
            try_time = 0
            try_max = np.random.randint(1, 6)
            print(try_max)
            while try_time < try_max:
                try_time += 1
                matching_indices = np.argwhere(
                    processed_organ_region == organ_standard_val)
                if matching_indices.size > 0:
                    random_index = np.random.choice(matching_indices.shape[0])
                    start_point = matching_indices[random_index]
                    large = np.random.randint(0,3)
                
                    processed_organ_region[start_point[0], start_point[1], start_point[2]] = threshold/2  # start point initialize
                    current_state[start_point[0], start_point[1], start_point[2]] = threshold/2
                    if large == 0:
                        x_offset = np.random.randint(-10,10)
                        y_offset = np.random.randint(-10,10)
                        z_offset = np.random.randint(-10,10)
                        if processed_organ_region[start_point[0] + x_offset, start_point[1] + y_offset, start_point[2] + z_offset] == organ_standard_val:
                            processed_organ_region[start_point[0] + x_offset, start_point[1] + y_offset, start_point[2] + z_offset] = threshold/2
                            current_state[start_point[0] + x_offset, start_point[1] + y_offset, start_point[2] + z_offset] = threshold/2
                            
                        x_offset = np.random.randint(-10,10)
                        y_offset = np.random.randint(-10,10)
                        z_offset = np.random.randint(-10,10)
                        if  processed_organ_region[start_point[0] + x_offset, start_point[1] + y_offset, start_point[2] + z_offset] == organ_standard_val:
                            processed_organ_region[start_point[0] + x_offset, start_point[1] + y_offset, start_point[2] + z_offset] = threshold/2
                            current_state[start_point[0] + x_offset, start_point[1] + y_offset, start_point[2] + z_offset] = threshold/2

                    print(start_point)
                current_state[start_point[0], start_point[1], start_point[2]] = threshold/2  # start point initialize
                
            all_states = []  # states of each step

            # simulate tumor growth
            tumor_out = grow_tumor(current_state, density_organ_state, kernel_size, steps, all_states,
                                organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map)

            # map to CT value
            print(cropped_img.dtype)
            step = 0
            while step<steps:
                step += 10 
                img_out = map_to_CT_value(cropped_img, tumor_out, density_organ_map,
                                        step, threshold, outrange_standard_val, organ_hu_lowerbound, organ_standard_val, start_point)
                save_name = os.path.basename(file) + '_' + str(i) + '_' + str(step) + '.nii.gz'
            
                # save the result
                img_save = img.copy()
                img_save[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = img_out
                save = sitk.GetImageFromArray(img_save)
                sitk.WriteImage(save, save_path + '/img/' +save_name)

                mask_save = np.zeros_like(img_save)
                mask_save[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = tumor_out[step]
                mask_save[mask_save > 0] = 1

                save = sitk.GetImageFromArray(mask_save)
                sitk.WriteImage(save, save_path + '/mask/' +save_name)

                save_list.append(save_name)

    with open( '../list/save_list.csv', 'w') as f:
        for item in save_list:
            f.write(item + '\n')
    

if __name__ == "__main__":
    main()