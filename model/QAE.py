import tensorflow as tf

from model.utils.layer import Quad_conv_layer_same
from model.utils.layer import Quad_conv_layer_valid
from model.utils.layer import Quad_deconv_layer_valid_linear
from model.utils.layer import Quad_deconv_layer_same
from model.utils.layer import Quad_deconv_layer_same_linear


def encoder(x_input,is_training=False,
	verbose=False):
	# encoder chunck
	encode_conv1 = Quad_conv_layer_same(
		x_input,shape=[3, 3, 1, 15])
	encode_conv2 = Quad_conv_layer_same(
		encode_conv1,shape=[3, 3,15,15])
	encode_conv3 = Quad_conv_layer_same(
		encode_conv2,shape=[3, 3,15,15])
	encode_conv4 = Quad_conv_layer_same(
		encode_conv3,shape=[3, 3,15,15])
	encode_conv5 = Quad_conv_layer_valid(
		encode_conv4,shape=[3, 3,15,15])

	layer_dict = {'encode_1':encode_conv1,
		'encode_2':encode_conv2,
		'encode_3':encode_conv3,
		'encode_4':encode_conv4,
		'encode_5':encode_conv5}

	return encode_conv5, layer_dict


def decoder(d_input,layer_dict,
	is_training=False,verbose=False):
	# decoder chunck
	decode_conv4 = tf.nn.relu(
		Quad_deconv_layer_valid_linear(
			d_input,shape=[3, 3,15,15],
			outputshape=tf.shape(
				layer_dict['encode_4']))+
		layer_dict['encode_4'])
	decode_conv3 = Quad_deconv_layer_same(
		decode_conv4,shape=[3, 3,15,15],
		outputshape=tf.shape(
			layer_dict['encode_3']))
	decode_conv2 = tf.nn.relu(
		Quad_deconv_layer_same_linear(
			decode_conv3,shape=[3, 3,15,15],
			outputshape=tf.shape(
				layer_dict['encode_2']))+
		layer_dict['encode_2'])
	decode_conv1 = Quad_deconv_layer_same(
		decode_conv2,shape=[3, 3,15,15],
		outputshape=tf.shape(
			layer_dict['encode_1']))

	return decode_conv1


def build_QAE(x_input,is_training=False,
	verbose=False):
	
	encoder_out, layer_dict = encoder(
		x_input=x_input,
		is_training=is_training,
		verbose=verbose)

	decoder_out = decoder(d_input=encoder_out,
		layer_dict=layer_dict,
		is_training=is_training,
		verbose=verbose)

	prediction = tf.nn.relu(
		Quad_deconv_layer_same_linear(
			decoder_out,shape=[3, 3,1,15],
			outputshape=tf.shape(x_input))+
		x_input)

	return prediction

# if __name__ == '__main__':
  
#     x = 255*tf.ones( shape = [32,300,300,1],
#     	dtype=tf.dtypes.float32, name=None)

#     result = build_QAE(x_input=x)

#     print(result.get_shape())

