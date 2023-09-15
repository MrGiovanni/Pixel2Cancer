from clara.viz.widgets import Widget
from clara.viz.core import Renderer
import clara.viz.core
import numpy as np
from clara.viz.core import DataDefinition
import SimpleITK as sitk
from PIL import Image


data_definition = DataDefinition('/home/ylai/code/Tumor_Growth/data/img/ct.nii.gz')


# show with PIL
image = Image.fromarray(renderer)
image.save("./image_save_20200509.jpg")
