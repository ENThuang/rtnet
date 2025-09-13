from .noise_augmentations import augment_gaussian_blur, augment_gaussian_noise
from .abstract_transforms import AbstractTransform
import numpy as np
from typing import Union, Tuple


class GaussianNoiseTransform(AbstractTransform):
    def __init__(
        self,
        noise_variance=(0, 0.1),
        p_per_sample=1,
        p_per_channel: float = 1,
        per_channel: bool = False,
        data_key="data",
    ):
        """
        Adds additive Gaussian Noise

        :param noise_variance: variance is uniformly sampled from that range
        :param p_per_sample:
        :param p_per_channel:
        :param per_channel: if True, each channel will get its own variance sampled from noise_variance
        :param data_key:

        CAREFUL: This transform will modify the value range of your data!
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gaussian_noise(
                    data_dict[self.data_key][b],
                    self.noise_variance,
                    self.p_per_channel,
                    self.per_channel,
                )
        return data_dict


class GaussianBlurTransform(AbstractTransform):
    def __init__(
        self,
        blur_sigma: Tuple[float, float] = (1, 5),
        different_sigma_per_channel: bool = True,
        different_sigma_per_axis: bool = False,
        p_isotropic: float = 0,
        p_per_channel: float = 1,
        p_per_sample: float = 1,
        data_key: str = "data",
    ):
        """

        :param blur_sigma:
        :param data_key:
        :param different_sigma_per_axis: if True, anisotropic kernels are possible
        :param p_isotropic: only applies if different_sigma_per_axis=True, p_isotropic is the proportion of isotropic
        kernels, the rest gets random sigma per axis
        :param different_sigma_per_channel: whether to sample a sigma for each channel or all channels at once
        :param p_per_channel: probability of applying gaussian blur for each channel. Default = 1 (all channels are
        blurred with prob 1)
        """
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.blur_sigma = blur_sigma
        self.different_sigma_per_axis = different_sigma_per_axis
        self.p_isotropic = p_isotropic

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gaussian_blur(
                    data_dict[self.data_key][b],
                    self.blur_sigma,
                    self.different_sigma_per_channel,
                    self.p_per_channel,
                    different_sigma_per_axis=self.different_sigma_per_axis,
                    p_isotropic=self.p_isotropic,
                )
        return data_dict
