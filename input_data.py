"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import gzip
import os
import urllib

import numpy as np


class DataSet(object):

  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data(directory):
    '''
    directory: including two folders: observations/ and normals/
    read in observations for pixels and the related normals
    return observations and normals
    '''
    npy_files = os.listdir(directory+'observations/')
    observation_collection = []
    normal_collection = []
    for filename in npy_files:
        model_index_string = filename.split('_')[0]
        normal_filename = directory+'normals/normal'+model_index_string+'.txt'
        normals = read_normal_to_array(normal_filename)
        if normal_collection == []:
            normal_collection = normals
        else:
            normal_collection = np.concatenate((normal_collection, normals))
        temp = np.load(directory+'observations/'+filename)
        if observation_collection == []:
            observation_collection = temp
        else:
            observation_collection = np.concatenate( (observation_collection, temp) )
    return observation_collection,normal_collection

def read_normal_to_array(normal_filename):
    '''
    read normals from text file, floats are splited by ' '
    '''
    with open(normal_filename) as fid:
        line = fid.readline()

    normal_strings = line.split()
    n_nums = []
    for string in normal_strings:
        n_nums.append(float(string))
    normals = np.reshape(n_nums,(-1,3))
    return normals

def read_data_sets(train_channel_index):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.validation = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets

  VALIDATION_SIZE = 5000
  # read data from specific directory
  train_observations = np.load('data/train/train_channel'+str(train_channel_index)+'.npy');
  train_normals = np.load('data/train/train_normals.npy');
  test_observations = np.load('data/test/test_channel'+str(train_channel_index)+'.npy');
  test_normals = np.load('data/test/test_normals.npy');
  # split training data into validation data and training data
  validation_images = train_observations[:VALIDATION_SIZE]
  validation_labels = train_normals[:VALIDATION_SIZE]
  train_images = train_observations[VALIDATION_SIZE:]
  train_labels = train_normals[VALIDATION_SIZE:]
  print("train:%s, validation:%s, test:%s"%(train_labels.shape, validation_labels.shape, test_normals.shape))
  # initialize data_sets with the splited data
  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_observations, test_normals)

  return data_sets
