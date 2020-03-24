import os
import tensorflow as tf
from load_dataset import load_data
from model import QAE
from model.utils.losses import l2_loss
from model.utils.losses import ssim_loss
from model.utils.losses import perceptual_loss
from model.augmentations.add_noise import gaussian_noise
from model.augmentations.add_noise import poisson_noise
from model.augmentations.add_noise import random_noise


config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


