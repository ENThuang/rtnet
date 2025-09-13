import numpy as np


class IntensityClip(object):
    def __init__(self, low_val=-400, high_val=400):
        self.low_val = low_val
        self.high_val = high_val

    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError(
                "'IntensityClip' expects to receive Numpy ndarry as input. %s is received"
                % type(data)
            )
        data = data.clip(self.low_val, self.high_val)

        return data
