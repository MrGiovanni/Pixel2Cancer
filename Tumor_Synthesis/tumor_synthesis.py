from typing import Hashable, Mapping, Dict
import random
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, RandomizableTransform

from .utils import generate_tumor,get_predefined_texture
import numpy as np


Organ_List = {'liver': [1,2], 'pancreas': [1,2], 'kidney': [1,2]}
Organ_HU = {'liver': [100, 160],'pancreas': [100, 160], 'kidney': [140, 200]}
steps = {'liver': 150, 'pancreas': 80, 'kidney': 80}
class TumorSysthesis(RandomizableTransform, MapTransform):
    def __init__(self, 
    keys: KeysCollection, 
    prob: float = 0.1,
    allow_missing_keys: bool = False,
    args: Dict = None,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        random.seed(0)
        np.random.seed(0)
        self.args = args
        self.organ_name = args.organ  # organ name
        self.steps = steps[args.organ]  # step
        self.kernel_size = (3, 3, 3)  # Receptive Field
        self.organ_hu_lowerbound = Organ_HU[args.organ][0]  # organ hu lowerbound
        self.outrange_standard_val = Organ_HU[args.organ][1]  # outrange standard value
        self.organ_standard_val = 0  # organ standard value
        self.threshold = 10  # threshold
        
        
        self.textures = []
        sigma_as = [3, 6, 9, 12, 15]
        sigma_bs = [4, 7]
        predefined_texture_shape = (500, 500, 500)
        for sigma_a in sigma_as:
            for sigma_b in sigma_bs:
                texture = get_predefined_texture(predefined_texture_shape, sigma_a, sigma_b)
                self.textures.append(texture)
        print("All predefined texture have generated.")



    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if self._do_transform and (np.max(d['label']) <= 2):
    
            texture = random.choice(self.textures)
            d['image'][0], d['label'][0] = generate_tumor(d['image'][0], d['label'][0],texture, self.steps, self.kernel_size, self.organ_standard_val, self.organ_hu_lowerbound, self.outrange_standard_val, self.threshold, self.organ_name, self.args)
            
        return d
