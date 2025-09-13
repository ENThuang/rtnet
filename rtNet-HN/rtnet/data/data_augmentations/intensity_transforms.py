from .abstract_transforms import AbstractTransform
from .intensity_augmentations import intensity_normalization


class IntensityNormalization(AbstractTransform):
    def __init__(
        self,
        normalization_schemes,
        use_nonzero_mask=None,
        data_key="data",
        mask_key="mask",
        intensityproperties_key="intensityproperties",
    ):
        self.data_key = data_key
        self.mask_key = mask_key
        self.intensityproperties_key = intensityproperties_key
        self.normalization_schemes = normalization_schemes
        self.use_nonzero_mask = use_nonzero_mask

    def __call__(self, **data_dict):
        data, mask = intensity_normalization(
            data_dict[self.data_key],
            data_dict[self.mask_key],
            data_dict[self.intensityproperties_key],
            self.normalization_schemes,
        )
        data_dict[self.data_key] = data
        data_dict[self.mask_key] = mask
        return data_dict
