import torch
from math import sqrt
from itertools import product


class PriorBox:
    def __init__(self, cfg):
        self.image_size = cfg.INPUT.IMAGE_SIZE
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale_x = self.image_size[0] / self.strides[k][0]
            scale_y = self.image_size[1] / self.strides[k][1]
            for i, j in product(range(f[1]), range(f[0])):
                # unit center x,y
                cx = (j + 0.5) / scale_x
                cy = (i + 0.5) / scale_y

                # small sized square box
                size_x = self.min_sizes[k][0]
                size_y = self.min_sizes[k][1]
                h = size_y / self.image_size[1]
                w = size_x / self.image_size[0]
                priors.append([cx, cy, w, h])

                # big sized square box
                size_x = sqrt(self.min_sizes[k][0] * self.max_sizes[k][0])
                size_y = sqrt(self.min_sizes[k][1] * self.max_sizes[k][1])
                h = size_y / self.image_size[1]
                w = size_x / self.image_size[0]
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size_x = self.min_sizes[k][0]
                size_y = self.min_sizes[k][1]
                h = size_y / self.image_size[1]
                w = size_x / self.image_size[0]
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
