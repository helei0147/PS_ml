import os,sys
import numpy as np

def main():
    train_channel1, train_channel2, train_channel3, train_normals = read_data('data/train/')
    test_channel1, test_channel2, test_channel3, test_normals = read_data('data/test/')
    # split training data into validation data and training data
    np.save('train_channel1.npy', train_channel1);
    np.save('train_channel2.npy', train_channel2);
    np.save('train_channel3.npy', train_channel3);
    np.save('train_normals.npy', train_normals);
    np.save('test_channel1.npy', test_channel1);
    np.save('test_channel2.npy', test_channel2);
    np.save('test_channel3.npy', test_channel3);
    np.save('test_normals.npy',test_normals);

def read_data(directory):
    '''
    directory: including two folders: observations/ and normals/
    read in observations for pixels and the related normals
    return observations and normals
    '''
    npy_files = os.listdir(directory+'observations/')
    normal_dict = read_all_normals()
    channel1_collection = []
    channel2_collection = []
    channel3_collection = []
    normal_collection = []
    for filename in npy_files:
        print(filename)
        model_index = int(filename.split('_')[0])
        normals = normal_dict[model_index]
        temp = np.load(directory+'observations/'+filename)
        temp_channel1 = temp[0::3,:]
        temp_channel2 = temp[1::3,:]
        temp_channel3 = temp[2::3,:]
        assert temp_channel1.shape[0] == normals.shape[0],("ob:%s nm:%s"%(temp.shape,normals.shape))
        if channel1_collection == []:
            channel1_collection = temp_channel1
            channel2_collection = temp_channel2
            channel3_collection = temp_channel3
        else:
            channel1_collection = np.concatenate((channel1_collection, temp_channel1))
            channel2_collection = np.concatenate((channel2_collection, temp_channel2))
            channel3_collection = np.concatenate((channel3_collection, temp_channel3))
        if normal_collection == []:
            normal_collection = normals
        else:
            normal_collection = np.concatenate( (normal_collection, normals) )
    return channel1_collection,channel2_collection,channel3_collection,normal_collection
def read_all_normals():
    normal_collection = []
    for i in range(10):
        normal_filename = '../normal/normal'+str(i)+'.txt'
        normals = read_normal_to_array(normal_filename)
        print(normals.shape[0])
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
if __name__ == '__main__':
    main()
