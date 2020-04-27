import tensorflow as tf

def bias_variable(shape):
    initial = tf.constant(-0.02, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d_valid(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_same(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d_valid(x,W,outputshape):
    return tf.nn.conv2d_transpose(x,W,output_shape=outputshape,strides= [1,1,1,1], padding = 'VALID')

def deconv2d_same(x,W,outputshape):
    return tf.nn.conv2d_transpose(x,W,output_shape=outputshape,strides= [1,1,1,1], padding = 'SAME')

def Quad_deconv_layer_valid_linear(input, shape, outputshape):
    
    W_r = weight_variable_Wr(shape)
    #W_g = weight_variable_Wg(shape)
    W_g = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    #W_b = weight_variable_Wb(shape)
    W_b = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    b_r = tf.Variable(tf.constant(0, shape=[shape[2]],dtype=tf.float32))
    b_g = tf.Variable(tf.constant(1, shape=[shape[2]],dtype=tf.float32))
    c = tf.Variable(tf.constant(0, shape=[shape[2]],dtype=tf.float32))
    return (deconv2d_valid(input, W_r, outputshape)+b_r)*(deconv2d_valid(input, W_g,outputshape)+b_g)+deconv2d_valid(input*input, W_b,outputshape)+c

def Quad_deconv_layer_same_linear(input, shape, outputshape):
    
    W_r = weight_variable_Wr(shape)
    #W_g = weight_variable_Wg(shape)
    W_g = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    #W_b = weight_variable_Wb(shape)
    W_b = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    b_r = tf.Variable(tf.constant(0, shape=[shape[2]],dtype=tf.float32))
    b_g = tf.Variable(tf.constant(1, shape=[shape[2]],dtype=tf.float32))
    c = tf.Variable(tf.constant(0, shape=[shape[2]],dtype=tf.float32))
    return (deconv2d_same(input, W_r, outputshape)+b_r)*(deconv2d_same(input, W_g,outputshape)+b_g)+deconv2d_same(input*input, W_b,outputshape)+c

def Quad_deconv_layer_same(input, shape, outputshape):
    
    W_r = weight_variable_Wr(shape)
    #W_g = weight_variable_Wg(shape)
    W_g = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    #W_b = weight_variable_Wb(shape)
    W_b = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    b_r = tf.Variable(tf.constant(0, shape=[shape[2]],dtype=tf.float32))
    b_g = tf.Variable(tf.constant(1, shape=[shape[2]],dtype=tf.float32))
    c = tf.Variable(tf.constant(0, shape=[shape[2]],dtype=tf.float32))
    return tf.nn.relu((deconv2d_same(input, W_r, outputshape)+b_r)*(deconv2d_same(input, W_g,outputshape)+b_g)+deconv2d_same(input*input, W_b,outputshape)+c)



def weight_variable_Wr(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial)

def weight_variable_Wg(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial)

def weight_variable_Wb(shape):
    initial = tf.truncated_normal(shape, stddev=0.01,dtype=tf.float32)
    return tf.Variable(initial)


def Quad_conv_layer_valid(input, shape):
    
    W_r = weight_variable_Wr(shape)
    #W_g = weight_variable_Wg(shape)
    W_g = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    #W_b = weight_variable_Wb(shape)
    W_b = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))  # W_b can also be initialized by Gaussian with samll variance
    b_r = tf.Variable(tf.constant(0, shape=[shape[3]],dtype=tf.float32))
    b_g = tf.Variable(tf.constant(1, shape=[shape[3]],dtype=tf.float32))
    c = tf.Variable(tf.constant(0, shape=[shape[3]],dtype=tf.float32))
    
    return tf.nn.relu((conv2d_valid(input, W_r)+b_r)*(conv2d_valid(input, W_g)+b_g)+conv2d_valid(input*input, W_b)+c) 

def Quad_conv_layer_same(input, shape):
    
    W_r = weight_variable_Wr(shape)
    #W_g = weight_variable_Wg(shape)
    W_g = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    #W_b = weight_variable_Wb(shape)
    W_b = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    b_r = tf.Variable(tf.constant(0, shape=[shape[3]],dtype=tf.float32))
    b_g = tf.Variable(tf.constant(1, shape=[shape[3]],dtype=tf.float32))
    c = tf.Variable(tf.constant(0, shape=[shape[3]],dtype=tf.float32))
    
    return tf.nn.relu((conv2d_same(input, W_r)+b_r)*(conv2d_same(input, W_g)+b_g)+conv2d_same(input*input, W_b)+c) 
