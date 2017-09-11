import os,sys
import numpy as np

def main(_):
    train_observations,train_normals = read_data('data/train/')
    test_observations, test_normals = read_data('data/test/')
    # split training data into validation data and training data
    np.save('train_observations.npy', train_observations);
    np.save('train_normals.npy',train_normals);
    np.save('test_observations.npy',test_observations);
    np.save('test_normals.npy',test_normals);

def read_data(directory):
    '''
    directory: including two folders: observations/ and normals/
    read in observations for pixels and the related normals
    return observations and normals
    '''
    npy_files = os.listdir(directory+'observations/')
    normal_dict = read_all_normals()
    observation_collection = []
    normal_collection = []
    for filename in npy_files:
        model_index = int(filename.split('_')[0])
        normals = normal_dict[model_index]
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
def read_all_normals():
    normal_collection = []
    for i in range(10):
        normal_filename = '../normal/normal'+str(i)'.txt'
        normals = read_normal_to_array(normal_filename)
        normal_collection.append(normals)
    return normal_collection
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
if '__name__' == '__main__':
    main()
