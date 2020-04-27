import tensorflow as tf
from pathlib import Path


def decode_img(img):
	img = tf.image.decode_png(img,channels=1)
	img = tf.image.convert_image_dtype(img,
		tf.uint8)
	return img

def process_path(file_path):
	img = tf.io.read_file(file_path)
	img = decode_img(img)
	return img

def load_set(img_path, set_type='train'):
	list_ds = tf.data.Dataset.list_files(
		str(img_path/'*.png'))
	dataset = list_ds.map(process_path,
		num_parallel_calls=
		tf.data.experimental.AUTOTUNE)
	return dataset

def load_data(dataset='mias',DIR = Path('.')):
	
	if dataset == 'mias':
		path = DIR/'mias_dataset'
	elif dataset == 'dental':
		path = DIR/'dental_dataset'
	else:
		path = DIR/'both_datasets'

	train_path = path/'train'
	val_path = path/'val'
	test_path = path/'test'

	train_ds = load_set(train_path,set_type='train')
	val_ds = load_set(val_path,set_type='val')
	test_ds = load_set(test_path,set_type='test')

	return train_ds,val_ds,test_ds

# def show_batch(image_batch):
# 	plt.figure(figsize=(10,10))
# 	for n in range(25):
# 		ax = plt.subplot(5,5,n+1)
# 		plt.imshow(image_batch[n].squeeze(),
# 			cmap='gray', vmin=0, vmax=255)
# 		plt.axis('off')
# 	plt.show()

# @tf.function
# def sample_batch(dataset):
# 	return next(iter(dataset))


# if __name__ =='__main__':

# 	import matplotlib.pyplot as plt

# 	train,_,test = load_data(dataset='dental')

# 	train_batch = train.batch(32)

# 	with tf.Session() as sess:
# 		image_batch = sample_batch(train_batch)
# 		show_batch(image_batch.eval())

