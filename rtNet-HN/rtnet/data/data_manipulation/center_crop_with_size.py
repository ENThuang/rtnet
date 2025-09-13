import numpy as np
from sqlite3 import DataError
from collections import OrderedDict
from rtnet.utils import find_bbox_from_mask, crop_and_pad


class CenterCropAndMaskOut(object):
    def __init__(
        self,
        final_patch_size,
        memmap_mode="r",
        pad_mode="constant",
        pad_kwargs_data=None,
        data_key="data",
        label_key="mask",
        is_2d=False,
    ):
        self.final_patch_size = final_patch_size
        self.memmap_mode = memmap_mode
        self.pad_mode = pad_mode
        self.data_key = data_key
        self.label_key = label_key
        self.is_2d = is_2d

        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data

    def __call__(self, **data_dict):
        raw_data = data_dict.get(self.data_key)  # (mod, chn, z, x, y)
        raw_mask = data_dict.get(self.label_key)
        dim = len(raw_data.shape)
        assert (
            dim == 4 if self.is_2d else 5
        ), "The dimension of the input data must be 5 for 3d \
                and 4 for 2d, but {} got".format(
            dim
        )

        # raw_data = np.squeeze(raw_data.copy(), axis=(0, 1))
        # if dim == 4:
        #     raw_mask = np.squeeze(raw_mask.copy(), axis=0)
        # else:
        #     raw_mask = np.squeeze(raw_mask.copy(), axis=1)
        # raw_mask = np.squeeze(raw_mask.copy(), axis=0)

        cur_patch_size = raw_data.shape[-3:] if not self.is_2d else raw_data.shape[-2:]
        image_data = np.copy(raw_data)
        mask = np.copy(raw_mask)

        cropped_masks = []
        cropped_data = []
        if not self.is_2d:
            if np.any(mask[0, 1] > 0):
                x_tl, y_tl, z_tl, h, w, d = find_bbox_from_mask(mask[0, 1])
                bbox_center = (x_tl + h // 2, y_tl + w // 2, z_tl + d // 2)
            else:
                pseudo_h, pseudo_w, pseudo_d = mask[0, 1].shape
                bbox_center = (pseudo_h // 2, pseudo_w // 2, pseudo_d // 2)
        else:
            x_tl, y_tl, h, w = find_bbox_from_mask(mask[0, 1])
            bbox_center = (x_tl + h // 2, y_tl + w // 2)

        for sample_id in range(image_data.shape[0]):
            temp_data = []
            for channel_id in range(image_data.shape[1]):
                cropped_image_data = crop_and_pad(
                    image_data[sample_id, channel_id],
                    cur_patch_size,
                    bbox_center,
                    self.final_patch_size,
                    pad_mode=self.pad_mode,
                    pad_kwargs_data=self.pad_kwargs_data,
                )

                temp_data.append(cropped_image_data)
            cropped_data.append(np.concatenate(temp_data, axis=0))
        cropped_data = np.stack(cropped_data, axis=0)

        for sample_id in range(mask.shape[0]):
            temp_mask = []
            for channel_id in range(mask.shape[1]):
                cropped_mask = crop_and_pad(
                    mask[sample_id, channel_id],
                    cur_patch_size,
                    bbox_center,
                    self.final_patch_size,
                    pad_mode=self.pad_mode,
                    pad_kwargs_data=self.pad_kwargs_data,
                )

                # cropped_mask = np.expand_dims(cropped_mask, axis=0)
                temp_mask.append(cropped_mask)
            cropped_masks.append(np.concatenate(temp_mask, axis=0))
        cropped_masks = np.stack(cropped_masks, axis=0)

        # if dim == 4:
        #     cropped_masks = np.concatenate(cropped_masks, axis=1)
        # else:
        #     cropped_masks = np.concatenate(cropped_masks, axis=0)

        return {self.data_key: cropped_data, self.label_key: cropped_masks}
