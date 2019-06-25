#!/usr/bin/env python 
# -*- coding:utf-8 -*-


import tensorflow as tf

def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[''], [''], [''], [''], [''], [''], ['']]
    lesion_id,image_id,dx,dx_type,age,sex,localization = tf.decode_csv(value, defaults)
    return image_id,dx

def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    image_id, dx = read_data(file_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    image_id_batch, dx_batch = tf.train.shuffle_batch(
        [image_id,dx], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return image_id_batch, dx_batch

x_train_batch, y_train_batch = create_pipeline('data/HAM10000_metadata.csv', 1)


init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()  # local variables like epoch_num, batch_size
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Retrieve a single instance:
    try:
        #while not coord.should_stop():
        while True:
            example, label = sess.run([x_train_batch, y_train_batch])
            print (example)
            print (label)
    except tf.errors.OutOfRangeError:
        print ('Done reading')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()