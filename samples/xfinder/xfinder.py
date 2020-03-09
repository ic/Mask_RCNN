"""
Based on: Matterport's implementation of Mask-RCNN sample code.
"""

import colorsys
import datetime
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw
import skimage.draw
from skimage.measure import find_contours

ROOT_DIR = os.path.abspath('../../')

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from loaders.via import Config, Dataset

# Path to base weight file (COCO)
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'mrcnn_coco.h5')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


def train(model, target, training_manifest, validation_manifest, dataset):

    dataset_train = Dataset(target)
    dataset_train.load_via(training_manifest, dataset)
    dataset_train.prepare()

    dataset_val = Dataset(target)
    dataset_val.load_via(validation_manifest, dataset)
    dataset_val.prepare()

    print('Training network heads only (enough for many tasks, and faster)')
    try:
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,
                    layers='heads')
    except StopIteration:
        print('Training over')
        pass


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


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    image = np.asarray(image).copy()
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return Image.fromarray(image)


def overlay(image, boxes, masks, class_ids, class_names,
            scores=None, title="",
            figsize=(16, 16), ax=None,
            show_mask=True, show_bbox=True,
            colors=None, captions=None):
    height, width = image.shape[:2]

    # Object count
    N = boxes.shape[0]
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = colors or random_colors(N)

    masked_image = Image.fromarray(image)
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        draw = ImageDraw.Draw(masked_image)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            draw.polygon(verts)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        draw.text((x1, y1 + 8), caption)
    return masked_image


def apply_to(model, target, image_path, out_path=None, masksonly=False, manifest=None):
    print('Running on {}'.format(image_path))

    image = skimage.io.imread(image_path)
    r = model.detect([image], verbose=1)[0]
    class_names = ['bg', target]
    res_img = overlay(image,
                      r['rois'], r['masks'], r['class_ids'],
                      class_names, r['scores'])

    ofname, _ = os.path.splitext(image_path)
    fname = '{}_out.png'.format(ofname)
    if out_path:
        fname = os.path.join(out_path, os.path.basename(fname))
    if not masksonly:
        res_img.save(fname)
        print('Saved to: ', fname)

    try:
        manf = None
        if manifest:
            manf = open(manifest, 'a')
        original_fname = os.path.basename(image_path)
        print(f"Will save {r['masks'].shape[-1]} masks")
        for idx in range(r['masks'].shape[-1]):
            mask = r['masks'][...,idx]
            if r['scores'][idx] > 0.9:
                m = Image.fromarray(mask)
            mname = f'{ofname}_mask_{idx}.jpg'
            m.save(mname)
            if manf:
                manf.write(','.join([original_fname, os.path.basename(mname) + '\n']))
    finally:
        if manifest:
            manf.close()


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
                        help="Please provide a path, or 'coco' to start from scratch or 'last'.")
    parser.add_argument('--training_manifest', required=False,
                        metavar='/path/to/manifest.json',
                        help='Manifest for the training dataset in VIA format.')
    parser.add_argument('--validation_manifest', required=False,
                        metavar='/path/to/manifest.json',
                        help='Manifest for the validation dataset in VIA format.')
    parser.add_argument('--dataset', required=False,
                        metavar='/path/to/dataset',
                        help='Directory of the dataset.')
    parser.add_argument('--image', required=False,
                        metavar='path',
                        help='Path or URL to image to infer on.')
    parser.add_argument('--outdir', '-o', required=False,
                        metavar='path',
                        help='Directory where to save the output to.')
    parser.add_argument('--masksonly', required=False,
                        action='store_true',
                        help='Output only the masks.')
    parser.add_argument('--manifest', required=False,
                        metavar='path',
                        help='Path to a CSV manifest file to report to.')
    args = parser.parse_args()

    # Validate arguments
    if args.command == 'train':
        assert args.dataset, 'Argument --dataset is required for training'
    elif args.command == 'infer':
        assert args.image, 'Provide --image to infer on image'

    print('Weights: ', args.weights)
    print('Dataset: ', args.dataset)
    print('Training Manifest: ', args.training_manifest)
    print('Validation Manifest: ', args.validation_manifest)
    print('Results: ', OUTPUT_DIR)

    # Configuration
    if args.command == 'train':
        config = Config(args.target)
    else:
        class InferenceConfig(Config):
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
        train(model, args.target, args.training_manifest, args.validation_manifest, args.dataset)
    elif args.command == 'infer':
        apply_to(model, args.target, args.image, args.outdir, args.masksonly, args.manifest)
