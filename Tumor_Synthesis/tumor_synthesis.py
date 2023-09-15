from typing import Hashable, Mapping, Dict
import random
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, RandomizableTransform

from .utils import generate_tumor
import numpy as np

Organ_List = {'liver': [1,2]}
Organ_HU = {'liver': [100, 160]}

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
        
        self.steps = 100  # step
        self.kernel_size = (3, 3, 3)  # Receptive Field
        self.organ_hu_lowerbound = Organ_HU['liver'][0]  # organ hu lowerbound
        self.outrange_standard_val = Organ_HU['liver'][1]  # outrange standard value
        self.organ_standard_val = 0  # organ standard value
        self.threshold = 10  # threshold
        self.organ_name = 'liver'  # organ name
        self.args = args



    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        if self._do_transform and (np.max(d['label']) <= 1):
            d['image'][0], d['label'][0] = generate_tumor(d['image'][0], d['label'][0], self.steps, self.kernel_size, self.organ_standard_val, self.organ_hu_lowerbound, self.outrange_standard_val, self.threshold, self.organ_name, self.args)
        
        return d