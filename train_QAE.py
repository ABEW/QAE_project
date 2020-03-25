import os
import tensorflow as tf
from pathlib import Path
from tensorflow.data import Iterator
from tensorflow.contrib import summary


from model import QAE
from load_dataset import load_data
from model.utils.losses import l2_loss
from model.utils.losses import ssim_loss
from model.utils.losses import perceptual_loss
from model.augmentations.add_noise import gaussian_noise
from model.augmentations.add_noise import poisson_noise
from model.augmentations.add_noise import random_noise

import warnings
warnings.filterwarnings("ignore")

tf.compat.v1.logging.set_verbosity(
	tf.compat.v1.logging.WARN)


config = tf.compat.v1.ConfigProto(
	log_device_placement=True)
config.gpu_options.allow_growth = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def training():

	batch_size = 32
	noise_type = 'gaussian'
	noise_proportion = 0.2
	noise_mean = 0
	noise_std = 1
	noise_lam = 1
	noise_std_range = [1,5]
	noise_lam_range = [1,5]
	loss_type = 'l2_loss'

	current_dir = Path('.')

	train_true,__,test_true=load_data(dataset='both',
		DIR=current_dir)

	train_true = train_true.repeat().batch(batch_size)
	train_true = train_true.prefetch(buffer_size =
		tf.data.experimental.AUTOTUNE)

	test_true = test_true.repeat().batch(batch_size)
	test_true = test_true.prefetch(buffer_size =
		tf.data.experimental.AUTOTUNE)

	iter_train_handle = train_true.make_one_shot_iterator().string_handle()
	iter_val_handle = test_true.make_one_shot_iterator().string_handle()

	handle = tf.placeholder(tf.string, shape=[])
	iterator = Iterator.from_string_handle(handle,
		train_true.output_types,
		train_true.output_shapes)

	next_batch = iterator.get_next()

	with tf.Session().as_default() as sess:
		global_step = tf.train.get_global_step()

		handle_train, handle_val = sess.run(
			[iter_train_handle, iter_val_handle])

		for step in range(5):

			train_true_img = sess.run(next_batch,
				feed_dict={handle: handle_train})

			test_true_img = sess.run(next_batch,
				feed_dict={handle: handle_val})

			train_true_img = tf.constant(train_true_img)
			test_true_img = tf.constant(test_true_img)

			if noise_type == 'random':
				train_noised = random_noise(
					train_true_img,proportion=
					noise_proportion,
					std_range=noise_std_range,
					lam_range=noise_lam_range)
				test_noised = random_noise(
					test_true_img,proportion=
					noise_proportion,
					std_range=noise_std_range,
					lam_range=noise_lam_range)
			elif noise_type == 'poisson':
				train_noised = poisson_noise(
					train_true_img,proportion=
					noise_proportion,
					lam = noise_lam)
				test_noised = poisson_noise(
					test_true_img,proportion=
					noise_proportion,
					lam = noise_lam)
			else:
				train_noised = gaussian_noise(
					train_true_img,proportion=
					noise_proportion,
					mean=noise_mean,
					std=noise_std)
				test_noised = gaussian_noise(
					test_true_img,proportion=
					noise_proportion,
					mean=noise_mean,
					std=noise_std)

			train_input = tf.cast(train_noised,
				dtype=tf.float32)

			test_input = tf.cast(test_noised,
				dtype=tf.float32)

			with tf.variable_scope('QAE'):
				train_denoised_img = QAE.build_QAE(
					train_input, is_training=True)

			with tf.variable_scope('QAE',reuse=tf.AUTO_REUSE):
				test_denoised_img = QAE.build_QAE(
					test_input, is_training=False)


			if loss_type == 'l2_loss':
				train_loss = l2_loss(
					tf.cast(train_true_img,
						dtype=tf.float32),
					train_denoised_img)
				val_loss = l2_loss(
					tf.cast(test_true_img,
						dtype=tf.float32),
					test_denoised_img)

		print('done')

if __name__ == '__main__':
	training()




