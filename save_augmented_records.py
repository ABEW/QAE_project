import cv2
import read_files
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from pathlib import Path

def rotate_90_degree(data):
	return np.rot90(data,k=1,axes=(-2,-1))

def rotate_180_degree(data):
	return np.rot90(data,k=2,axes=(-2,-1))

def flip_up_down(data):
	return np.flip(data,axis=-2)

def flip_left_right(data):
	return np.flip(data,axis=-1)

def augment_images(data):
	flip_ud = flip_up_down(data)
	flip_lr = flip_up_down(data)
	rotate_90 = rotate_90_degree(data)
	rotate_180 = rotate_180_degree(data)

	augmented_data = np.concatenate(
		[data,flip_ud,flip_lr,rotate_90,
		rotate_180],axis=0)

	np.take(augmented_data,
		np.random.permutation(augmented_data.shape[0]),
		axis=0,out=augmented_data)


	return augmented_data


def save_true_images(data,IMGDIR,\
	set_type = 'train'):

	shape = data.shape

	set_path = Path(IMGDIR/set_type)
	set_path.mkdir(parents=True,\
		exist_ok=True)

	if len(shape)==2:
		data = \
			data.expand_dims(data,axis=0)
	
	batch_size = data.shape[0]

	for i in range(batch_size):
		file_name = set_type+'_'+str(i)+\
			'.png'
		cv2.imwrite(\
			str(set_path/file_name),\
			data[i])

def load_true_images(IMGDIR=Path('.'),\
	mias=True,dental=True,train_set=300,
	val_set=0,augment=False):

	train_data,val_data,test_data=\
		read_files.load_dataset(IMGDIR,
			mias=mias, dental=dental,
			train_set=train_set,
			val_set=val_set)

	if augment:
		train_data=augment_images(train_data)


	return train_data, val_data, test_data

def create_all_datasets(train_set,
	val_set,augment=False):

	original_folder = Path('.')

	sets = ['train', 'val', 'test']

	# Mias only
	Dataset_1 = load_true_images(\
		IMGDIR = original_folder,\
		mias=True, dental=False,
		train_set=train_set,
		val_set=val_set,
		augment=augment)

	DS1_path = Path('mias_dataset_augmented')
	DS1_path.mkdir(parents=True,\
		exist_ok=True)

	# Dental only
	Dataset_2 = load_true_images(\
		IMGDIR = original_folder,\
		mias=False, dental=True,
		train_set=train_set,
		val_set=val_set,
		augment=augment)

	DS2_path = Path('dental_dataset_augmented')
	DS2_path.mkdir(parents=True,\
		exist_ok=True)

	# Both datasets
	Dataset_3 = load_true_images(\
		IMGDIR = original_folder,\
		mias=True, dental=True,
		train_set=train_set,
		val_set=val_set,
		augment=augment)

	DS3_path = Path('both_datasets_augmented')
	DS3_path.mkdir(parents=True,\
		exist_ok=True)

	Datasets =[Dataset_1, Dataset_2,\
		Dataset_3]

	Paths = [DS1_path, DS2_path,\
		DS3_path]

	for i in range(3):
		for j in range(3):
			save_true_images(\
				Datasets[i][j],\
				IMGDIR=Paths[i],\
				set_type=sets[j])

if __name__ == '__main__':

	create_all_datasets(train_set=150,
		val_set=100,augment=True)



