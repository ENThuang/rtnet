from warnings import warn

from .abstract_transforms import AbstractTransform
from .resample_augmentations import augment_linear_downsampling_scipy
import numpy as np

from rtnet.data.data_loader.data_image_preprocessing.preprocessing import (
    get_do_separate_z,
    get_lowres_axis,
    resample_data_or_seg,
)


class SimulateLowResolutionTransform(AbstractTransform):
    """Downsamples each sample (linearly) by a random factor and upsamples to original resolution again
    (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel:

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:
    """

    def __init__(
        self,
        zoom_range=(0.5, 1),
        per_channel=False,
        p_per_channel=1,
        channels=None,
        order_downsample=1,
        order_upsample=0,
        data_key="data",
        p_per_sample=1,
        ignore_axes=None,
    ):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_linear_downsampling_scipy(
                    data_dict[self.data_key][b],
                    zoom_range=self.zoom_range,
                    per_channel=self.per_channel,
                    p_per_channel=self.p_per_channel,
                    channels=self.channels,
                    order_downsample=self.order_downsample,
                    order_upsample=self.order_upsample,
                    ignore_axes=self.ignore_axes,
                )
        return data_dict


class ResampleTransform(SimulateLowResolutionTransform):
    def __init__(
        self,
        zoom_range=(0.5, 1),
        per_channel=False,
        p_per_channel=1,
        channels=None,
        order_downsample=1,
        order_upsample=0,
        data_key="data",
        p_per_sample=1,
    ):
        warn(
            "This class is deprecated. It was renamed to SimulateLowResolutionTransform. Please change your code",
            DeprecationWarning,
        )
        super(ResampleTransform, self).__init__(
            zoom_range,
            per_channel,
            p_per_channel,
            channels,
            order_downsample,
            order_upsample,
            data_key,
            p_per_sample,
        )


class ResamlePixelSpacing(AbstractTransform):
    def __init__(
        self,
        dataset_properties,
        anisotropy_threshold=3,
        target_spacing_percentile=50,
        data_key="data",
        mask_key="mask",
        properties_key="properties",
    ):
        self.data_key = data_key
        self.mask_key = mask_key
        self.properties_key = properties_key
        self.dataset_properties = dataset_properties
        self.target_spacing_percentile = target_spacing_percentile
        self.anisotropy_threshold = anisotropy_threshold

        self.order_data = 3
        self.order_mask = 1
        self.order_z_data = 0
        self.order_z_mask = 0

        self.target_spacing = self.get_target_spacing()

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        mask = data_dict[self.mask_key]
        if len(data.shape) == 5:
            data = np.squeeze(data, axis=0)
        if len(mask.shape) == 5:
            mask = np.squeeze(mask, axis=0)

        original_spacing = np.array(data_dict[self.properties_key]["original_spacing"])
        target_spacing = self.target_spacing

        assert not ((data is None) and (mask is None))
        if data is not None:
            assert len(data.shape) == 4, "data must be c x y z"
        if mask is not None:
            assert len(mask.shape) == 4, "seg must be c x y z"

        shape = np.array(data[0].shape)
        new_shape = np.round(
            (
                (np.array(original_spacing) / np.array(target_spacing)).astype(float)
                * shape
            )
        ).astype(int)

        # temporal solution to avoid resampling in the z direction from a thinner CT
        if new_shape[0] < shape[0]:
            new_shape[0] = shape[0]
            print("Warning: not resampling in the z direction from a thinner CT")

        if get_do_separate_z(original_spacing, self.anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, self.anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

        if axis is not None:
            if len(axis) == 3:
                # every axis has the spacing, this should never happen, why is this code here?
                do_separate_z = False
            elif len(axis) == 2:
                # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
                # separately in the out of plane axis
                do_separate_z = False
            else:
                pass

        data_reshaped = np.zeros(
            tuple([data.shape[0]] + list(new_shape)), dtype=data.dtype
        )
        data_reshaped[:, ::-1] = resample_data_or_seg(
            data[:, ::-1],
            new_shape,
            False,
            axis,
            self.order_data,
            do_separate_z,
            order_z=self.order_z_data,
            verbose=False,
        )
        mask_reshaped = np.zeros(
            tuple([mask.shape[0]] + list(new_shape)), dtype=data.dtype
        )
        mask_reshaped[:, ::-1] = resample_data_or_seg(
            mask[:, ::-1],
            new_shape,
            True,
            axis,
            self.order_mask,
            do_separate_z,
            order_z=self.order_z_mask,
            verbose=False,
        )

        data_dict[self.data_key] = np.expand_dims(data_reshaped, axis=0)
        data_dict[self.mask_key] = np.expand_dims(mask_reshaped, axis=0)
        return data_dict

    def get_target_spacing(self):
        """
        per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
        and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

        For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
        (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
        resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
        impact performance (due to the low number of slices).
        """
        spacings = self.dataset_properties["all_spacings"]
        sizes = self.dataset_properties["all_sizes"]

        target = np.percentile(np.vstack(spacings), self.target_spacing_percentile, 0)
        target_size = np.percentile(np.vstack(sizes), self.target_spacing_percentile, 0)

        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (
            self.anisotropy_threshold * max(other_spacings)
        )
        has_aniso_voxels = target_size[
            worst_spacing_axis
        ] * self.anisotropy_threshold < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = (
                    max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
                )
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target
