"""
Usage:
    # Create train record:
    python generate_tfrecord.py \
        --csv_input=data/dataTrain.csv \
        --img_folder=images \
        --output_file=train.record
    # Create validation record:
    python generate_tfrecord.py \
        --csv_input=data/dataValidation.csv \
        --img_folder=images \
        --output_file=validation.record
    # Create test record:
    python generate_tfrecord.py \
        --csv_input=data/dataTest.csv  \
        --img_folder=images \
        --output_file=test.record
"""
from __future__ import division, print_function

import argparse
import io
import json
import sys

import pandas as pd
import tensorflow as tf
from PIL import Image

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _create_tf_example(data):
    """
    create TFRecord example from a single row of data.
    """
    # Encode jpg file.
    img_url = data['jpg']
    img_name = data['jpg'].split('.')[-2]

    # Read encoded image file, and get properties we need.
    with tf.gfile.GFile(img_url, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = img_name.encode('utf8')
    image_format = b'jpg'

    # Encode json file.
    json_url = data['json']
    with open(json_url) as json_file:
        points = json.load(json_file)

    # After getting all the features, time to generate a TensorFlow example.
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/filename': _bytes_feature(filename),
        'image/source_id': _bytes_feature(filename),
        'image/encoded': _bytes_feature(encoded_jpg),
        'image/format': _bytes_feature(image_format),
        'label/points': _float32_feature(points),
    }))
    return tf_example


def main(unused_argv):
    """
    entrance
    """
    tf_writer = tf.python_io.TFRecordWriter(FLAGS.output_file)
    examples = pd.read_csv(FLAGS.csv_input)
    for _, row in examples.iterrows():
        current_example = _create_tf_example(row)
        tf_writer.write(current_example.SerializeToString())
    tf_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv_input',
        type=str,
        default='data/dataValidation.csv',
        help='Directory where the data file is.'
    )
    parser.add_argument(
        '--img_folder',
        type=str,
        default="dataValidation",
        help="Directory where the images live."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="dataValidation.record",
        help="Where the record file should be placed."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

