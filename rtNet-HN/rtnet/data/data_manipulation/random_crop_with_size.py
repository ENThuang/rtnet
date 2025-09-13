import numpy as np
from collections import OrderedDict


class RandomCropWithSize(object):
    def __init__(
        self, final_patch_size, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None
    ):
        self.final_patch_size = final_patch_size
        self.memmap_mode = memmap_mode
        self.pad_mode = pad_mode

        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data

    def __call__(self, data):
        if len(data.shape) == 3:
            cur_patch_size = data.shape
        else:
            cur_patch_size = data.shape[1:]

        need_to_pad = (
            np.array(cur_patch_size) - np.array(self.final_patch_size)
        ).astype(int)
        for d in range(3):
            # if data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] < 0:
                if need_to_pad[d] + data.shape[d + 1] < self.final_patch_size[d]:
                    need_to_pad[d] = self.final_patch_size[d] - data.shape[d + 1]
            else:
                need_to_pad[d] = 0

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        shape = data.shape[1:]
        if need_to_pad[0] > 0:
            lb_x = -need_to_pad[0] // 2
            ub_x = (
                shape[0]
                + need_to_pad[0] // 2
                + need_to_pad[0] % 2
                - self.final_patch_size[0]
            )
        else:
            lb_x = -self.final_patch_size[0] // 2
            ub_x = shape[0] - self.final_patch_size[0] // 2
        if need_to_pad[1] > 0:
            lb_y = -need_to_pad[1] // 2
            ub_y = (
                shape[1]
                + need_to_pad[1] // 2
                + need_to_pad[1] % 2
                - self.final_patch_size[1]
            )
        else:
            lb_y = -self.final_patch_size[1] // 2
            ub_y = shape[1] - self.final_patch_size[1] // 2
        if need_to_pad[2] > 0:
            lb_z = -need_to_pad[2] // 2
            ub_z = (
                shape[2]
                + need_to_pad[2] // 2
                + need_to_pad[2] % 2
                - self.final_patch_size[2]
            )
        else:
            lb_z = -self.final_patch_size[2] // 2
            ub_z = shape[2] - self.final_patch_size[2] // 2

        # sample the bbox randomly from lb and ub
        bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
        bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
        bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

        bbox_x_ub = bbox_x_lb + self.final_patch_size[0]
        bbox_y_ub = bbox_y_lb + self.final_patch_size[1]
        bbox_z_ub = bbox_z_lb + self.final_patch_size[2]

        # we first crop the data to the region of the bbox that actually lies within the data.
        # This will result in a smaller array which is then faster to pad. Valid_bbox is just
        # the coord that lied within the data cube. It will be padded to match the patch size
        # later
        valid_bbox_x_lb = max(0, bbox_x_lb)
        valid_bbox_x_ub = min(shape[0], bbox_x_ub)
        valid_bbox_y_lb = max(0, bbox_y_lb)
        valid_bbox_y_ub = min(shape[1], bbox_y_ub)
        valid_bbox_z_lb = max(0, bbox_z_lb)
        valid_bbox_z_ub = min(shape[2], bbox_z_ub)

        # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
        # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
        # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
        # remove label -1 in the data augmentation but this way it is less error prone)
        cropped_data = np.copy(
            data[
                :,
                valid_bbox_x_lb:valid_bbox_x_ub,
                valid_bbox_y_lb:valid_bbox_y_ub,
                valid_bbox_z_lb:valid_bbox_z_ub,
            ]
        )

        cropped_data = np.pad(
            cropped_data,
            (
                (0, 0),
                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0)),
            ),
            self.pad_mode,
            **self.pad_kwargs_data
        )

        if cropped_data.shape != (1, 32, 224, 224):
            a = 10

        return cropped_data
