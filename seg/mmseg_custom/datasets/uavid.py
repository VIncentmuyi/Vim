# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class UAVidDataset(CustomDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('Background clutter', 'Building', 'Road', 'Static car', 'Tree', 'Low vegetation','Human', 'Moving car')

    PALETTE = [[0,0,0], [128,0,0], [128,64,128], [0,128,0],
               [128,128,0], [64,0,128], [192,0,192], [64,64,0]]

    def __init__(self, **kwargs):
        super(UAVidDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)