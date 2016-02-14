import os
from lasagne.utils import floatX
import matplotlib.pyplot as plt
import numpy as np

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def extract_all(image_paths, prefix, prepare_img_fun, cnn_feature_fun):
    chunk_size = 64
    cnn_features = []
    for chunk_idx, image_paths_chunk in enumerate(chunks(image_paths, chunk_size)):
        print 'extracting chunk {}/{}...'.format(chunk_idx + 1, int(len(image_paths)/chunk_size) + 1)
        cnn_inputs = floatX(np.zeros((len(image_paths_chunk), 3 , 224, 224)))
        
        for idx, image_path in enumerate(image_paths_chunk):
            abs_img_path = os.path.join(prefix, image_path)
            try:
                im = plt.imread(abs_img_path)
                _, cnn_inputs[idx] = prepare_img_fun(im)
            except IOError as exp:
                print exp
                continue
        cnn_features_chunk = cnn_feature_fun(cnn_inputs)
        cnn_features.append(cnn_features_chunk)
        print 'extracting chunk {}/{}...DONE'.format(chunk_idx+1, int(len(image_paths)/chunk_size)+1)
    return np.concatenate(cnn_features)

