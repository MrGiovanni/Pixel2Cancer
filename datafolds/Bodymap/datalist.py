import numpy as np
import os
import json
import random
import shutil
import nibabel as nib







# read datalist from txt file, and write into json file
datadict = {
    "description": "all healthy data",
    "labels": {
        "0": "background",
        "1": "organ",
        "2": "tumor"
    },
    "licence": "yt",
    "modality": {
        "0": "CT"
    },
    "name": "kidney",
    "numTraining": 0,
    "numValid": 0,
    "reference": "xxxx",
    "release": "2024/01/01",
    "tensorImageSize": "3D",
    "training": [],
    "validation": []
}
organ = "kidney"
tumor = "kidney_tumor"
# read list from txt

with open('AbdomenAtlas1.1.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        datadict['training'].append({"image": line + "/ct.nii.gz", "label": line + "/segmentations/" + organ + ".nii.gz", "tumor": line + "/segmentations/" + tumor + ".nii.gz"})

# separate training and validation in 5 fold
random.shuffle(datadict['training'])
fold = 5
length = len(datadict['training'])
for i in range(fold):
    start = int(i * length / fold)
    end = int((i + 1) * length / fold)
    datadict['validation'] = datadict['training'][start:end]
    datadict['training'] = datadict['training'][0:start] + datadict['training'][end:]
    # save to json
    with open('Atlas9k_kidney_fold' + str(i) + '.json', 'w') as f:
        json.dump(datadict, f, indent=4)
        print("write to datalist.json")

 