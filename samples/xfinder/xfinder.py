"""
Based on: Matterport's implementation of Mask-RCNN sample code.
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from mrcnn import visualize

ROOT_DIR = os.path.abspath('../../')

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to base weight file (COCO)
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'mrcnn_coco.h5')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


############################################################
#  Configuration
############################################################

class XfinderConfig(Config):

    def __init__(self, name):
        # mrcnn requires a configuration name.
        self.NAME = name

    IMAGES_PER_GPU = 2 # A GPU with 12Gb memory fits about two images.

    NUM_CLASSES = 1 + 1  # Background + item to find

    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class ViaDataset(utils.Dataset):

    def __init__(self, category):
        self.category = category

    def load_via(self, manifest, dataset_dir, subset):
        """
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        self.add_class(self.category, 1, self.category)

        # Train or validation dataset?
        assert subset in ['train', 'val']
        dataset_dir = os.path.join(dataset_dir, subset)

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
        annotations = json.load(open(os.path.join(dataset_dir, manifest)))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
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


def train(model, target, manifest, dataset):
    """Train the model."""
    # Training dataset.
    dataset_train = ViaDataset(target)
    dataset_train.load_via(manifest, dataset, 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ViaDataset(target)
    dataset_val.load_via(manifest, dataset, 'val')
    dataset_val.prepare()

    print('Training network heads only (enough for many tasks, and faster)')
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
      image: RGB image [height, width, 3]
      mask: instance segmentation mask [height, width, instance count]

      Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def apply_to(model, target, image_path):
    print('Running on {}'.format(image_path))

    image = skimage.io.imread(image_path)
    r = model.detect([image], verbose=1)[0]
    class_names = ['bg', target]
    res_img = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'],
                                title='Predictions')

    fname, _ = os.path.splitext(image_path)
    file_name = '{}_out.png'.format(fname)
    res_img.savefig(file_name)

    print('Saved to: ', file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to find something.')
    parser.add_argument('command',
                        metavar='<command>',
                        choices=['train', 'infer'],
                        help="Select 'train' or 'infer'.")
    parser.add_argument('target',
                        metavar='<target>',
                        help='The target object we want to train on (e.g. balloons, cats, dogs, elephant).')
    parser.add_argument('weights',
                        metavar='<weights>',
                        choices=['coco', 'last'],
                        help="Select 'coco' to start from scratch or 'last'.")
    parser.add_argument('--manifest', required=False,
                        metavar='/path/to/manifest.json',
                        help='Manifest for the dataset in VIA format.')
    parser.add_argument('--dataset', required=False,
                        metavar='/path/to/dataset',
                        help='Directory of the dataset.')
    parser.add_argument('--image', required=False,
                        metavar='path',
                        help='Path or URL to image to infer on.')
    args = parser.parse_args()

    # Validate arguments
    if args.command == 'train':
        assert args.dataset, 'Argument --dataset is required for training'
    elif args.command == 'infer':
        assert args.image, 'Provide --image to infer on image'

    print('Weights: ', args.weights)
    print('Dataset: ', args.dataset)
    print('Manifest: ', args.manifest)
    print('Results: ', OUTPUT_DIR)

    # Configuration
    if args.command == 'train':
        config = ViaConfig(args.target)
    else:
        class InferenceConfig(ViaConfig):
            # Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig(args.target)
    config.display()

    # Model
    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config,
                                  model_dir=OUTPUT_DIR)
    else:
        model = modellib.MaskRCNN(mode='inference', config=config,
                                  model_dir=OUTPUT_DIR)

    # Weights file
    if args.weights.lower() == 'coco':
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            # Download weights file if not present.
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == 'last':
        weights_path = model.find_last()
    else:
        weights_path = args.weights

    print('Loading weights: ', weights_path)
    if args.weights.lower() == 'coco':
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            'mrcnn_class_logits', 'mrcnn_bbox_fc',
            'mrcnn_bbox', 'mrcnn_mask'])
    else:
        model.load_weights(weights_path, by_name=True)

    # Run mode
    if args.command == 'train':
        train(model, args.target, args.manifest, args.dataset)
    elif args.command == 'infer':
        apply_to(model, args.target, args.image)
