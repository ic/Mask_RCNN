import json
import os
import sys

import numpy as np
import skimage.draw

ROOT_DIR = os.path.abspath('../../../')
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

class Config(Config):

    def __init__(self, name):
        super().__init__()
        # mrcnn requires a configuration name.
        self.NAME = name

    IMAGES_PER_GPU = 2 # A GPU with 12Gb memory fits about two images.

    NUM_CLASSES = 1 + 1  # Background + item to find

    STEPS_PER_EPOCH = 20
    DETECTION_MIN_CONFIDENCE = 0.9


class Dataset(utils.Dataset):

    def __init__(self, category):
        super().__init__()
        self.category = category

    def load(self, manifest, dataset_dir):
        """
        dataset_dir: Root directory of the dataset.
        """
        self.add_class(self.category, 1, self.category)

        for d in json.load(open(manifest)):
            if len(d['annotations']) == 0:
                continue

            self.add_image(
                self.category,
                image_id=d['name'], # use file name as a unique image id
                path=os.path.join(dataset_dir, d['filename']),
                width=d['width'],
                height=d['height'],
                polygons=d['annotations'])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
          masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
          class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a target dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info['source'] != self.category:
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], len(info['polygons'])],
                        dtype=np.uint8)
        for i, polygon in enumerate(info['polygons']):
            # Get indexes of pixels inside the polygon and set them to 1
            xs, ys = list(zip(*polygon))
            rr, cc = skimage.draw.polygon(ys, xs)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info['source'] == self.category:
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)
