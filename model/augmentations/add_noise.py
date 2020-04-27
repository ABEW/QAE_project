import tensorflow as tf

def gaussian_noise(image,proportion=0.2,
	mean=0, std=1):
    '''Apply gaussina noise of given mean,
    	standard deviation and proportion
    	or scale of the noise
    Args:
        image: '3D Tensor' [batch, height, width]
        proportion: scalar - scale of noise
        mean: scalar
        std: scalar - standard deviation
    Returns:
        '3D Tensor' of same dtype and shape
    '''

    noise = proportion*tf.random.normal(
        shape=image.get_shape().as_list(),
        mean=mean,
        stddev=std,
        dtype=tf.float32)
    image = tf.cast(
        image,
        tf.float32) / 255.0 + noise

    # image = tf.cast(
    #     (
    #         tf.clip_by_value(
    #             image,
    #             0.0,
    #             1.0)
    #         * 255.0),
    #     tf.uint8)

    img_min = tf.math.reduce_min(image)
    img_max = tf.math.reduce_max(image)

    img_range = img_max - img_min

    image = tf.cast(
    	255*(image - img_min)/img_range,
    	tf.uint8)

    return image

def poisson_noise(image,proportion=0.2,
	lam=1):
    '''Apply poisson noise of given lambda,
    	and proportion/scale of the noise
    	Implemented as additive noise
    Args:
        image: '3D Tensor' [batch, height, width]
        proportion: scalar - scale of noise
        lam: scalar - Expected value of poisson
    Returns:
        '3D Tensor' of same dtype and shape
    '''

    noise = proportion*tf.random.poisson(
        shape=image.get_shape().as_list(),
        lam=lam, dtype=tf.float32)
    image = tf.cast(
        image,
        tf.float32) / 255.0 + noise

    # image = tf.cast(
    #     (
    #         tf.clip_by_value(
    #             image,
    #             0.0,
    #             1.0)
    #         * 255.0),
    #     tf.uint8)
    img_min = tf.math.reduce_min(image)
    img_max = tf.math.reduce_max(image)

    img_range = img_max - img_min

    image = tf.cast(
    	255*(image - img_min)/img_range,
    	tf.uint8)

    return image

def random_noise(image,proportion=0.2,
	std_range=[1,5], lam_range=[1,5]):
	'''Randomly apply gaussian or poisson noise 
		with stdev and lambda selected randomly 
		from the range
    Args:
        image: '3D Tensor' [batch, height, width]
        proportion: scalar - scale of noise
        std_range: '2D list of int'
        lam_range: '2D list of int'
    Returns:
        '3D Tensor' of same dtype and shape
    '''

	lam = tf.cast(
		tf.random.shuffle(tf.constant(\
		list(range(lam_range[0],lam_range[1])))),
		tf.float32)

	std = tf.cast(
		tf.random.shuffle(tf.constant(\
		list(range(std_range[0],std_range[1])))),
		tf.float32)

    # conditionally add noise
	image = tf.cond(tf.random.uniform(
            shape=[],
            dtype=tf.float32) > 0.5,
        lambda: gaussian_noise(image,\
        	proportion=proportion, std=std[0]),
        lambda: poisson_noise(image,\
        	proportion=proportion, lam=lam[0]))
    
	return image

# if __name__ == '__main__':
  
#     x = 255*tf.ones( shape = [3,100,100],
#     	dtype=tf.dtypes.uint8, name=None)
#     y = tf.ones( shape = [3,100,100],
#     	dtype=tf.dtypes.float32, name=None)

#     with tf.Session() as sess:
#     	x_original = sess.run(x)
#     	x_gauss = sess.run(gaussian_noise(x,
#     		std=2))
#     	x_poisson = sess.run(poisson_noise(x,
#     		lam=1))
#     	x_random = sess.run(random_noise(x))

#     	fig, axes = plt.subplots(nrows=1,ncols=4,
#     		figsize=(10,10))
#     	axes[0].imshow(x_original[0],cmap='gray',
#     		vmin=0, vmax=255)
#     	axes[1].imshow(x_gauss[0],cmap='gray',
#     		vmin=0, vmax=255)
#     	axes[2].imshow(x_poisson[0],cmap='gray',
#     		vmin=0, vmax=255)
#     	axes[3].imshow(x_random[0],cmap='gray',
#     		vmin=0, vmax=255)

#     	plt.show()
