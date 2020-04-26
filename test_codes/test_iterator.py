import tensorflow as tf
from tensorflow.data import Dataset, Iterator

dataset_train = Dataset.range(10)
dataset_val = Dataset.range(90, 100)

iter_train_handle = dataset_train.make_one_shot_iterator().string_handle()
iter_val_handle = dataset_val.make_one_shot_iterator().string_handle()

handle = tf.placeholder(tf.string, shape=[])
iterator = Iterator.from_string_handle(
    handle, dataset_train.output_types, dataset_train.output_shapes)
next_batch = iterator.get_next()

with tf.train.MonitoredTrainingSession() as sess:
    handle_train, handle_val = sess.run([iter_train_handle, iter_val_handle])

    for step in range(10):
        print('train', sess.run(next_batch, feed_dict={handle: handle_train}))

        if step % 3 == 0:
            print('val', sess.run(next_batch, feed_dict={handle: handle_val}))