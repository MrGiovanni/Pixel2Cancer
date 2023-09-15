import os
import SimpleITK as sitk
# Get the list of data files
def get_tumor_list():
    folder_path = '/ccvl/net/ccvl15/zzhou82/LargePseudoDataset/04_LiTS'

    # Get the list of files in the folder
    folder_list = os.listdir(folder_path)
    folder_list.sort()
    tumor_list = []
    healthy_list = []
    for i in folder_list:
        print(i)
        file_name = folder_path + '/' + i + '/original_label.nii.gz'
        mask = sitk.ReadImage(file_name)
        mask = sitk.GetArrayFromImage(mask)
        if 2 in mask:
            tumor_list.append(folder_path + '/' + i)
        else:
            healthy_list.append(folder_path + '/' + i)
    return tumor_list, healthy_list

if __name__ == '__main__':
    tumor_list, healthy_list = get_tumor_list()
    #write to csv
    with open('./list/tumor_list.csv', 'w') as f:
        for i in tumor_list:
            f.write(i + '\n')
    with open('./list/healthy_list.csv', 'w') as f:
        for i in healthy_list:
            f.write(i + '\n')
