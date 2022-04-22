#!/usr/bin/env python

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to download all datasets and create .tfrecord files.
"""

import base64
import collections
import pickle
import gzip
import os
import tarfile
import tempfile
from urllib import request
from matplotlib.cbook import flatten

import numpy as np
import scipy.io
from absl import app
import cv2
import random
from itertools import islice
from PIL import Image as Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

input_size = 224

def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def _encode_png(images):
    raw = []
    for image in images:
        to_png = base64.b64encode(image)
        raw.append(to_png)
    return raw


def _decode_png(encoded_str):
    bytes_str = base64.decodebytes(encoded_str)
    flattened = np.frombuffer(bytes_str, dtype=np.uint8)
    return flattened
    

def _load_redtheme():
    '''
    Add a function to acquire the dataset to scripts/create_datasets.py similar to the present ones, e.g. _load_cifar10.
    You need to call _encode_png to create encoded strings from the original images.
    The created function should return a dictionary of the format {
        'train' : {'images': <encoded 4D NHWC list>, 'labels': <1D int array>},
        'test'  : {'images': <encoded 4D NHWC list>, 'labels': <1D int array>}}
    :return:
    '''
    train_dir = '/opt/chenhaoqing/data/redtheme/train'
    test_dir = '/opt/chenhaoqing/data/redtheme/test'
    train_set, test_set = {}, {}
    train_lst, test_lst = [], []
    train_labels, test_labels = [], []

    # load, resize, bgr->rgb
    cv_resize = lambda x: np.array(Image.open(x).resize((input_size,input_size)))
    for rt, dirs, files in os.walk(train_dir):
        for f in files:
            if os.path.splitext(f)[-1] in ['.jpg', '.png', '.jpeg']:
                train_lst.append(os.path.join(rt, f))
    random.shuffle(train_lst)

    for img in train_lst:
        cls_dir = os.path.basename(os.path.dirname(img))
        assert cls_dir in ['0_正常', '1_红色头像', '2_红色宣传画'], f"illegal cls {cls_dir}"
        label = int(cls_dir.split('_')[0])
        assert label in [0, 1, 2], f"illegal label {label}"
        if cv_resize(img).shape == (input_size, input_size, 3):
            train_labels.append(label)
    train_set['labels'] = np.asarray(train_labels)
    train_set['data'] = _encode_png(np.reshape([cv_resize(i) for i in train_lst if cv_resize(i).shape == (input_size, input_size, 3)], (-1, input_size, input_size, 3)))

    for rt, dirs, files in os.walk(test_dir):
        for f in files:
            if os.path.splitext(f)[-1] in ['.jpg', '.png', '.jpeg']:
                test_lst.append(os.path.join(rt, f))
    random.shuffle(test_lst)

    for img in test_lst:
        cls_dir = os.path.basename(os.path.dirname(img))
        assert cls_dir in ['0_正常', '1_红色头像', '2_红色宣传画'], f"illegal cls {cls_dir}"
        label = int(cls_dir.split('_')[0])
        assert label in [0, 1, 2], f"illegal label {label}"
        if cv_resize(img).shape == (input_size, input_size, 3):
            test_labels.append(label)
    test_set['labels'] = np.asarray(test_labels)
    test_set['data'] = _encode_png(np.reshape([cv_resize(i) for i in test_lst if cv_resize(i).shape == (input_size, input_size, 3)], (-1, input_size, input_size, 3)))

    return dict(train=train_set, test=test_set)


def _load_redtheme_unlabel():
    '''
    Add a function to acquire the dataset to scripts/create_datasets.py similar to the present ones, e.g. _load_cifar10.
    You need to call _encode_png to create encoded strings from the original images.
    The created function should return a dictionary of the format {
        'train' : {'images': <encoded 4D NHWC list>, 'labels': <1D int array>},
        'test'  : {'images': <encoded 4D NHWC list>, 'labels': <1D int array>}}
    :return:
    '''
    unlabel_dir = '/opt/chenhaoqing/data/redtheme/unlabel'
    unlabel_set, unlabel_lst = {}, []
    unlabel_labels = []
    cv_resize = lambda x: np.array(Image.open(x).resize((input_size,input_size))) # cv2.resize(cv2.imread(x), (input_size, input_size))
    for rt, dirs, files in os.walk(unlabel_dir):
        for f in files:
            if os.path.splitext(f)[-1] in ['.jpg', '.png', '.jpeg']:
                unlabel_lst.append(os.path.join(rt, f))
    random.shuffle(unlabel_lst)

    for img in unlabel_lst:
        cls_dir = os.path.basename(os.path.dirname(img))
        assert cls_dir in ['0_正常', '1_红色头像', '2_红色宣传画'], f"illegal cls {cls_dir}"
        label = int(cls_dir.split('_')[0])
        assert label in [0, 1, 2], f"illegal label {label}"
        if cv_resize(img).shape == (input_size, input_size, 3):
            unlabel_labels.append(label)
    unlabel_set['labels'] = np.asarray(unlabel_labels)
    unlabel_set['data'] = _encode_png(np.reshape([cv_resize(i) for i in unlabel_lst if cv_resize(i).shape == (input_size, input_size, 3)], (-1, input_size, input_size, 3)))

    return dict(train=unlabel_set)


CONFIGS = dict(
    redtheme=dict(loader=_load_redtheme, checksums=dict(train=None, test=None)),
    redtheme_unlabel=dict(loader=_load_redtheme_unlabel, checksums=dict(train=None, test=None)),
)


def main(argv):
    if len(argv[1:]):
        subset = set(argv[1:])
    else:
        subset = set(CONFIGS.keys())
    # tf.gfile.MakeDirs(libml_data.DATA_DIR)
    for name, config in CONFIGS.items():
        if name not in subset:
            continue
        print('Preparing', name)
        datas = config['loader']() # 通过loader参数调用加载数据的方法
        if name.endswith('unlabel'):
            for i, item in enumerate(chunks(datas['train'], 15000)):
                with open('unlabel_batch_{}'.format(str(i)),"wb") as f:
                    pickle.dump(item,f)
        #labelled list
        else:
            # train list
            for i, item in enumerate(chunks(datas['train'], 15000)):
                with open('train_batch_{}'.format(str(i)),"wb") as f:
                    pickle.dump(item,f)
            # val list
            for i, item in enumerate(chunks(datas['test'], 15000)):
                with open('val_batch_{}'.format(str(i)),"wb") as f:
                    pickle.dump(item,f)


if __name__ == '__main__':
    app.run(main)
