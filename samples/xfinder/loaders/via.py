import json
import os
import sys

import numpy as np
import skimage.draw
import skimage.io

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
        super(Dataset, self).__init__()
        self.category = category

    def load(self, manifest, dataset_dir):
        """
        dataset_dir: Root directory of the dataset.
        """
        self.add_class(self.category, 1, self.category)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(manifest))
        annotations = annotations['_via_img_metadata']
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if 'regions' in a]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                shape_attributes = a['regions'].values()
            else:
                shape_attributes = a['regions']
            polygons = [r['shape_attributes'] for r in shape_attributes if 'shape_attributes' in r]

            # Ignore images that only contain background.
            if len(polygons) == 0:
                continue

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This does not scale well.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                self.category,
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

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
        for i, p in enumerate(info['polygons']):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
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
