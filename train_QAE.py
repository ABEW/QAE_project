import os
import tensorflow as tf
from pathlib import Path
from tensorflow.data import Iterator
from tensorflow.contrib import summary

import matplotlib.pyplot as plt


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

	batch_size = 10
	noise_type = 'gaussian'
	noise_proportion = 0.2
	noise_mean = 0
	noise_std = 1
	noise_lam = 1
	noise_std_range = [1,5]
	noise_lam_range = [1,5]
	loss_type = 'l2_loss'
	study_rate = 1e-5

	current_dir = Path('.')

	train_true,__,test_true=load_data(dataset='mias',
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

	noise_args = {'proportion':noise_proportion}

	if noise_type == 'random':
		noise_fn = random_noise
		noise_args['std_range'] = noise_std_range
		noise_args['lam_range'] = noise_lam_range
	elif noise_type == 'poisson':
		noise_fn = poisson_noise
		noise_args['lam'] = noise_lam
	else:
		noise_fn = gaussian_noise
		noise_args['mean'] = noise_mean
		noise_args['std'] = noise_std


	true_img = tf.placeholder(tf.uint8, 
		shape=[batch_size, 64, 64, 1])

	noised_img = noise_fn(**noise_args,
		image=true_img)

	model_input = tf.cast(noised_img,
		dtype=tf.float32)

	denoised_img = QAE.build_QAE(model_input)

	if loss_type == 'l2_loss':
		train_loss = l2_loss(
			tf.cast(true_img,
				dtype=tf.float32),
			denoised_img)
		# val_loss = l2_loss(
		# 	tf.cast(test_true_img,
		# 		dtype=tf.float32),
		# 	test_denoised_img)

	total_train_loss = train_loss

	optimizer = tf.train.AdamOptimizer(\
		learning_rate=study_rate).minimize(
		total_train_loss)

	tf.summary.scalar('train_l2_loss',train_loss)
	# tf.summary.scalar('val_l2_loss',val_loss)


	tf.summary.scalar('total_train_loss',
		total_train_loss)

	merged_summary = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(
		current_dir/'train_data')


	init_vars = tf.group(
		tf.global_variables_initializer(),
		tf.local_variables_initializer())

	saver = tf.train.Saver()


	with tf.Session().as_default() as sess:
		global_step = tf.train.get_global_step()

		handle_train, handle_val = sess.run(
			[iter_train_handle, iter_val_handle])

		sess.run(init_vars)

		for step in range(500):
			train_true_img = sess.run(next_batch,
				feed_dict={handle: handle_train})
			test_true_img = sess.run(next_batch,
				feed_dict={handle: handle_val})

			_ = sess.run(optimizer, 
				feed_dict={true_img:train_true_img})

			t_summ = sess.run(merged_summary,
				feed_dict={true_img:train_true_img})

			t_loss = sess.run(total_train_loss,
				feed_dict={true_img:train_true_img})

			train_writer.add_summary(t_summ,step)

			print('Iter:{}, Training Loss {}'.format(
				step, t_loss))

			if step%20 == 0:
				fig,axes = 

		print('done')

if __name__ == '__main__':
	training()



