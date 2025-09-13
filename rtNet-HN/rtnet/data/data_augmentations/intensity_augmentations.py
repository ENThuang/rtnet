import numpy as np


def intensity_normalization(
    data, mask, intensityproperties, normalization_schemes, use_nonzero_mask=None
):
    # assert len(normalization_schemes) == len(data), "normalization_schemes " \
    #                                                 "must have as many entries as data has " \
    #                                                 "modalities"
    assert use_nonzero_mask is None or len(use_nonzero_mask) == len(
        data
    ), "use_nonzero_mask must have as many entries as data has modalities"

    if len(data.shape) == 5:
        data = np.squeeze(data, axis=0)
    if len(mask.shape) == 5:
        mask = np.squeeze(mask, axis=0)

    for c in range(len(data)):
        scheme = normalization_schemes[c]
        if scheme == "CT":
            # clip to lb and ub from train data foreground and use foreground mn and sd from training data
            assert (
                intensityproperties is not None
            ), "ERROR: if there is a CT then we need intensity properties"
            mean_intensity = intensityproperties[c]["mean"]
            std_intensity = intensityproperties[c]["sd"]
            lower_bound = intensityproperties[c]["percentile_00_5"]
            upper_bound = intensityproperties[c]["percentile_99_5"]
            data[c] = np.clip(data[c], lower_bound, upper_bound)
            data[c] = (data[c] - mean_intensity) / std_intensity
            if use_nonzero_mask and use_nonzero_mask[c]:
                data[c][mask[-1] < 0] = 0
        elif scheme == "CT2":
            # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
            assert (
                intensityproperties is not None
            ), "ERROR: if there is a CT then we need intensity properties"
            lower_bound = intensityproperties[c]["percentile_00_5"]
            upper_bound = intensityproperties[c]["percentile_99_5"]
            mask = (data[c] > lower_bound) & (data[c] < upper_bound)
            data[c] = np.clip(data[c], lower_bound, upper_bound)
            mn = data[c][mask].mean()
            sd = data[c][mask].std()
            data[c] = (data[c] - mn) / sd
            if use_nonzero_mask[c]:
                data[c][mask[-1] < 0] = 0
        elif scheme in ["noNorm", "support", "heatmap"]:
            pass
        else:
            if use_nonzero_mask[c]:
                mask = mask[-1] >= 0
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (
                    data[c][mask].std() + 1e-8
                )
                data[c][mask == 0] = 0
            else:
                mn = data[c].mean()
                std = data[c].std()
                # print(data[c].shape, data[c].dtype, mn, std)
                data[c] = (data[c] - mn) / (std + 1e-8)
        data = np.expand_dims(data, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return data, mask
