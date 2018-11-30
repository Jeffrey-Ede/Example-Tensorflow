from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" #Specify GPUs visible to the process

import argparse

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

import time

import os, random

from PIL import Image #Useful for saving float32 images

import functools
import itertools

import collections
import six

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import run_config

from tensorflow.contrib.framework.python.ops import add_arg_scope

slim = tf.contrib.slim 

"""
This example script demonstrates how to train neural networks using 
TensorFlow version 1.7.0. It should work with other versions of 
TensorFlow; however, some minor adjustments may be needed.

TensorFlow can be installed with pip or another package manager e.g.
pip install tensorflow-gpu==1.7.0 to install version 1.7.0.

Other packages that are imported at the top of this script may not be 
essential in this script; however, they are generally useful. They are 
included so this script will throw errors for any packages that you
don't have so you know to install them.

In this script, we showcase a network that removes salt and pepper noise
(white and black speckles) from images. Most of the code in this script is 
boilerplate and is unlikey to need to be changed for other networks.

Written by: Jeffrey M. Ede
DD/MM/YYYY: 28/11/2018
Email: j.m.ede@warwick.ac.uk
"""

## State where resources are and which resources can be used at the top of the script

model_save_period = 1. #Save every this many hours
model_save_period *= 3600 #Convert to s

example_size = [256, 256] #Size of images the network will process
channels = 1 #Greyscale images have 1 channel

#Number of pixels in examples
example_px = 1
for x in example_size:
    example_px *= x

#Amount of salt and pepper noise (white and black speckles) to add to examples. 
#Pepper applied after and overrides salt
salt_prop = 0.03; pepper_prop = 0.08 

data_dir = "my/train/val/test/directories/are/located/here/"
if data_dir[-1] != '/':
    data_dir += '/'

model_dir = "save/my/network/and/training/outputs/here/"
if model_dir[-1] != '/':
    model_dir += '/'

log_file = model_dir+"log.txt"

shuffle_buffer_size = 5000
num_parallel_calls = 4
num_parallel_readers = 4
num_intra_threads = 0
prefetch_buffer_size = 10 #Prefetch up to this many examples from data processing pipeline
batch_size = 1
num_gpus = 1

num_epochs = 100000000000 #Dataset effetively repeats indefinitely

#Output example application of the neural network this often
save_result_every_n_batches = 5000

#Monitor performance on validation once for every this many training iterations
val_skip_n = 10

## Building blocks for neural networks 

def conv2d(input, num_channels, kernel_size=3, stride=1, padding="SAME", transpositional=False):
    """
    Convolution where weights, biases and ReLU activation are all set up
    automatically
    input: tensor to convolute
    num_channels: number of sets of filter kernels to apply to the channels
    of the input tensor. This is the number of output channels
    kernel_size: Size of kernels to learn to convolute accross the channels
    of the input tensor
    stride: Spatial stride to take between applications of kernels. A stride
    of 2 will half the output tensor's spatial size
    padding: What to do when kernels go partially over the edges of tensors
    when they acting on values at their sides
    transpotional: Backward pass of convolution used for upsampling. If this
    is true and stride is 2, the output tensor will double in spatial size
    """

    if not transpositional:
        x = tf.contrib.layers.conv2d(
                inputs=input,
                num_outputs=num_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                data_format="NHWC",
                rate=1,
                activation_fn=tf.nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None
            )
    else:
        x = tf.contrib.layers.conv2d_transpose(
                inputs=input,
                num_outputs=num_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                data_format="NHWC",
                activation_fn=tf.nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None
            )

    return x

def skip_2_residual_block(input, num_channels, kernel_size=3, padding='SAME'):
    """
    A skip-2 residual block consists of two successive convolutions and 
    a residual connection that adds the input tensor to the output of the
    convolutions
    input: tensor to convolute
    num_channels: number of sets of filter kernels to apply to the channels
    of the input tensor. This is the number of output channels
    kernel_size: Size of kernels to learn to convolute accross the channels
    of the input tensor
    padding: What to do when kernels go partially over the edges of tensors
    when they acting on values at their sides
    """

    x = conv2d(input, num_channels=num_channels, kernel_size=kernel_size, padding=padding)
    x = conv2d(x, num_channels=num_channels, kernel_size=kernel_size, padding=padding)

    x += input

    return x


## Create neural network graph from building blocks

def network(input, reuse=False):
    """
    This simple neural network is designed to translate images corrupted by 
    salt and pepper noise to images that are not corrupted. 
    It is designed to showcase 
    - Downsampling using strided convolution to decrease spatial size, 
      reducing calculations
    - Upsampling using transpositional strided convolution to increase spatial
      size
    - Skip-2 residual blocks, a popular building block used for processing in 
      neural networks used for image translation
    input: A batch of corrupted images to restore
    reuse: whether to reuse variables from another copy of this network that
    has been graphed
    """

    #It's good practice to put each important part of your network in its own variable scope
    #in case you want to reuse it with the same trainable variables (reuse=True) or to get
    #a list of trainable variables in a scope to apply an optimizer to
    with tf.variable_scope("Main_Network", reuse=reuse):

        #First convolution learns large 7x7 kernels, developing features using a large
        #neigbourhood of surrounding pixels
        x = conv2d(input=input, num_channels=32, kernel_size=7)

        #Striding reduces spatial size, reducing number of calculations needed. Since
        #the spatial size is smaller, we can develop more feature channels without such a high
        #impact on performance
        x = conv2d(input=x, num_channels=64, kernel_size=3, stride=2)

        #Let's use 4 skip-2 residual blocks for processing
        num_skip_2_residual_blocks = 4
        for _ in range(num_skip_2_residual_blocks):
            x = skip_2_residual_block(input=x, num_channels=64, kernel_size=3)
    
        #Upsample back to output size using a transpositional convolution. Since
        #the spatial size is larger, we reduce the number of feature channels to
        #reduce the impact on performance
        x = conv2d(input=x, num_channels=32, kernel_size=3, stride=2, transpositional=True)

        #Finally, we develop our output image. Since micrographs are greyscale, this only
        #has 1 output channel. We use a 7x7 kernel so the network can use a lot of information to make
        #its final decision. 
        x = conv2d(input=input, num_channels=1, kernel_size=7)

    return x


## Define the experiment that will be performed. That is, how the network(s) will be configured,
## losses that will be used to train them and training operations

def experiment(example_input, example_output, learning_rate, beta1):
    """
    This describes the configuration of neural networks and how they will
    be trained. 
    example_input: Batch of examples to be processed by the neural network(s).
    example_output: Batch of corresponding examples that the network is to
    be trained to output
    learning_rate: Scale loss by this amount during training
    beta1: decay rate used in running mean to calculate the first moment of 
    the momentum

    Returns: A dictionary containing performance statistics that can be used
    to monitor the neural network's progress, its output and training operations
    """
    
    #Restore examples
    restoration = network(example_input)

    #Calculate loss. Here we use a simple mean squared difference between
    #the restored images and our desired outputs
    loss = tf.losses.mean_squared_error(restoration, example_output)

    #Get parameters to train 
    model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Main_Network")

    #Optimize using the ADAM solver. This is the most popular momentum-based
    #solver for deep learning. It will often be important to experiment with the
    #learning rate and, in some cases, the first moment of the momentum (beta1).
    #As you get more experience, you will get a feel for good values for learning
    #rates and other parameters
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    train_op = optimizer.minimize(loss, var_list=model_params)

    #Return outputs in a dictionary. Being able to access them by key can save
    #a lot of time later if you decide to return more information from the
    #experiment
    return { 'restoration': restoration,
             'loss': loss,
             'train_op': train_op
             }

## Set up data pipeline. Rather than loading and preprocessing data as it is needed, it
## is more efficient to do this in parallel, storing prepared examples in a buffer. 
## Part of the pipeline is any custom processing functions used to augment data or
## prepare examples

def flip_rotate(img):
    """
    Applies a random combination of flips and 90 degree rotations to an image, leaving 
    the image unchanged 1/8 of the time. This augments the dataset by a factor of 8.
    """

    choice = np.random.randint(0, 8)
    
    if choice == 0:
        return img
    elif choice == 1:
        return np.rot90(img, 1)
    elif choice == 2:
        return np.rot90(img, 2)
    elif choice == 3:
        return np.rot90(img, 3)
    elif choice == 4:
        return np.flip(img, 0)
    elif choice == 5:
        return np.flip(img, 1)
    elif choice == 6:
        return np.flip(np.rot90(img, 1), 0)
    elif choice == 7:
        return np.flip(np.rot90(img, 1), 1)


def load_image(addr, img_type=np.float32):
    """
    Read a float32 image using its address. In case of failure, 
    return an image filled with zeros
    """
    
    try:
        img = imread(addr, mode='F')
    except:
        img = np.zeros(example_size)
        print("Image read failed")

    return img.astype(img_type)

def scale0to1(img):
    """
    Rescale image to be in [0,1]
    """

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img

def preprocess(img):
    """
    Replace any non-finite elements of image and scale it to have values in [0,1]
    """

    img[np.isnan(img)] = 0.
    img[np.isinf(img)] = 0.

    #Resize image if necessary, so that all examples have the same size
    if img.shape != example_size:
        img = cv2.resize(img, tuple(example_size), interpolation=cv2.INTER_AREA)

    img = scale0to1(img)

    return img.astype(np.float32)

def gen_lq(img):
    """
    Create low quality example that network will learn to create high-quality example from
    """

    lq = np.copy(img)

    #Add noise based on random number values
    rand_numbers = np.random.rand(example_size[0], example_size[1])

    lq[rand_numbers > 1-salt_prop] = 1. #Salt
    lq[rand_numbers < pepper_prop] = 0. #Pepper
    
    return lq.astype(np.float32)

def record_parser(record):
    """Parse files and generate lower quality images from them"""

    img = flip_rotate(preprocess(load_image(record)))
    lq = gen_lq(img)
    if np.sum(np.isfinite(img)) != example_px or np.sum(np.isfinite(lq)) != example_px:
        img = lq = np.copy(blank)

    return lq, img

def reshaper(img1, img2):
    """
    Function to be mapped accross dataset to ensure tensors yielded by dataset
    iterators have the correct shapes
    """

    img1 = tf.reshape(img1, example_size+[channels])
    img2 = tf.reshape(img2, example_size+[channels])
    return img1, img2

def input_fn(dir, subset, batch_size, num_shards):
    """
    Create a dataset from a list of filenames and shard batches from it for each GPU.
    Returns iterators that generate examples
    """

    with tf.device('/cpu:0'):

        #Create dataset as a list of filenames
        dataset = tf.data.Dataset.list_files(dir+subset+"/"+"*.tif")

        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)

        #Map the file reading and data preparation function across the dataset of 
        #filenames to translate the names to images
        dataset = dataset.map(
            lambda file: tf.py_func(record_parser, [file], [tf.float32, tf.float32]),
            num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(reshaper, num_parallel_calls=num_parallel_calls)

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

        #Prepare iterator that will keep going through the dataset
        iter = dataset.make_one_shot_iterator()
        img_batch = iter.get_next()

        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            return [img_batch[0]], [img_batch[1]]
        else: 
            #Split batch into shards if there are multiple GPUs
            image_batch = tf.unstack(img_batch, num=batch_size, axis=1)
            feature_shards = [[] for i in range(num_shards)]
            feature_shards_truth = [[] for i in range(num_shards)]
            for i in range(batch_size):
                idx = i % num_shards
                tensors = tf.unstack(image_batch[i], num=2, axis=0)
                feature_shards[idx].append(tensors[0])
                feature_shards_truth[idx].append(tensors[1])
            feature_shards = [tf.parallel_stack(x) for x in feature_shards]
            feature_shards_truth = [tf.parallel_stack(x) for x in feature_shards_truth]

            return feature_shards, feature_shards_truth

## The main function orchestrates the training, describing the flow of information to
## the network, to and from the hard drive and any buffer. It also coordinates training
## operations and typically provides a live print-out of performance statistics. Utility 
## functions, e.g. to display images, that might help with debugging, etc. can be placed 
## before it

def disp(img):
    """
    Scales image to lie in [0,1] then displays it.
    """

    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)

    return

class RunConfig(tf.contrib.learn.RunConfig): 
    def uid(self, whitelist=None):
        """
        Generates a 'Unique Identifier' based on all internal fields.
        Caller should use the uid string to check `RunConfig` instance integrity
        in one session use, but should not rely on the implementation details, which
        is subject to change.
        Args:
          whitelist: A list of the string names of the properties uid should not
            include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
            includes most properties user allowes to change.
        Returns:
          A uid string.
        """
        if whitelist is None:
            whitelist = run_config._DEFAULT_UID_WHITE_LIST

        state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        # Pop out the keys in whitelist.
        for k in whitelist:
            state.pop('_' + k, None)

        ordered_state = collections.OrderedDict(
            sorted(state.items(), key=lambda t: t[0]))
        # For class instance without __repr__, some special cares are required.
        # Otherwise, the object address will be used.
        if '_cluster_spec' in ordered_state:
            ordered_state['_cluster_spec'] = collections.OrderedDict(
                sorted(ordered_state['_cluster_spec'].as_dict().items(), key=lambda t: t[0]))
        return ', '.join(
            '%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state))

def main():
    """
    Manages data from pipeline, inputting it to neural network(s), coordinates 
    training, outputs performance statistics and provides a live-print out of 
    network performance.
    """

    tf.reset_default_graph()
    initialized_variables = set(tf.all_variables())

    with open(log_file, 'a') as log:
        log.flush()

        # The env variable is on deprecation path, default is set to off.
        os.environ['TF_SYNC_ON_FINISH'] = '0'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        #with tf.device("/cpu:0"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
        with tf.control_dependencies(update_ops):

            # Session configuration.
            log_device_placement = False #Once placement is correct, this fills up too much of the cmd window...
            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=log_device_placement,
                intra_op_parallelism_threads=num_intra_threads,
                gpu_options=tf.GPUOptions(force_gpu_compatible=True))

            config = RunConfig(
                session_config=sess_config, model_dir=model_dir)

            #Training iterators
            example_input, example_output = input_fn(data_dir, 'train', batch_size, num_gpus)
            #Validation iterators
            example_input_val, example_output_val = input_fn(data_dir, 'val', batch_size, num_gpus)

            with tf.Session(config=sess_config) as sess:

                print("Session started")

                #TensorFlow is written in C++ so TensorFlow variables must be initialized for use
                sess.run(tf.initialize_variables(set(tf.all_variables()) - initialized_variables))
                initialized_variables = set(tf.all_variables())

                #We are going to feed data from the data pipeline to placeholders in the graph
                example_input_for_making_ph, example_output_for_making_ph = sess.run([example_input, example_output])

                example_input_ph = [tf.placeholder(tf.float32, shape=i.shape) for i in example_input_for_making_ph]
                example_output_ph = [tf.placeholder(tf.float32, shape=i.shape) for i in example_output_for_making_ph]

                del example_input_for_making_ph, example_output_for_making_ph

                print("Dataflow established")

                #Learning hyperparameters
                learning_rate_ph = tf.placeholder(tf.float32)
                beta1_ph = tf.placeholder(tf.float32, shape=())

                #Create experiment. Tensors are returned in a dictionary so that they are easy to access by key
                exp_dict = experiment(example_input_ph[0], example_output_ph[0], learning_rate_ph, beta1_ph)

                #Group outputs into sets. This doesn't make sense when the experiment only returns a couple
                #of tensors; however, it is likely to save time if the experiment returns many values
                train_ops_names = ['train_op']
                train_ops = [exp_dict[x] for x in train_ops_names]

                output_ops_names = ['restoration']
                output_ops = [exp_dict[x] for x in output_ops_names]

                monitoring_ops_names = ['loss']
                monitoring_ops = [exp_dict[x] for x in monitoring_ops_names]

                print("Created experiment")

                #TensorFlow is written and implemented in C++ so TensorFlow variables must be initialized 
                #for use!
                sess.run( tf.initialize_variables( set(tf.all_variables())-initialized_variables), 
                          feed_dict={beta1_ph: np.float32(0.9)} )

                #Outputs file that can be viewed with TensorBoard. There is a memory leak for TensorFlow 1.7.0.
                train_writer = tf.summary.FileWriter( model_dir, sess.graph )

                #Only keep models from last 2 checkpoints saved. This saves the ENTIRE session, so save 
                #files can be quite big - a couple of GB in some cases. Deleting older models avoids 
                #wasting space
                saver = tf.train.Saver(max_to_keep=2)

                #Uncomment the line below to restore a saved session. Training will continue from the
                #checkpoint as usual, although you might want to update the counter variable below
                #saver.restore(sess, tf.train.latest_checkpoint(model_dir+"model/"))

                counter = 0 #Tracks number of training iterations
                val_counter = 0
                save_counter = counter
                counter_init = counter+1

                #From experience, this is a sensible learning rate to start training with here
                base_learning_rate = 0.01 
                #0.9 is the goto value for the first moment of the momentum decay coefficient. It's a good choice
                #for almost all networks
                beta1 = 0.9 

                #Total number of training iterations
                total_iters = 100_000

                #Number of steps to use when decaying the learning rate
                num_steps_in_lr_decay = 5

                print("Started training")

                while True: #A quit() call at the end of training will break the loop

                    #Monitor time to save every time the specified save time period elapses
                    time0 = time.time()
                    while time.time()-time0 < model_save_period:

                        validating = bool(val_counter % val_skip_n)
                        if not validating:
                            val_counter = 0
                        val_counter += 1

                        if val_counter % val_skip_n: #Only increment counter on non-validation iterations
                            counter += 1

                        #Implement a decaying learning rate schedule. Arbitrarilly, we will use the base 
                        #learning rate for the first half of training, then stepwise linearly decay
                        #the learning rate to zero. Stepwise; rather than continuous, decay helps 
                        #prevent overfitting
                        if counter < total_iters/2:
                            learning_rate = base_learning_rate
                        elif counter < total_iters:
                            rel_iters = counter - total_iters/2
                            decay_iters = total_iters/2
                            step = int(num_steps_in_lr_decay*rel_iters/decay_iters)
                            learning_rate = base_learning_rate * (1 - step/num_steps_in_lr_decay)
                        else:
                            saver.save(sess, save_path=model_dir, global_step=counter)
                            quit()

                        #Get example from data pipeline
                        _example_input, _example_output = sess.run([example_input, example_output])

                        #Assign values to placeholders in a dictionary, ready to flow into
                        #the graph. Importantly, everthing has to be numpy, rather than pure
                        #python variables.
                        feed_dict = { learning_rate_ph: np.float32(learning_rate),
                                      beta1_ph: np.float32(beta1),
                                      example_input_ph[0]: _example_input[0],
                                      example_output_ph[0]: _example_output[0]
                                    }

                        #Save outputs occasionally. Defaults to saving once every 1000 iterations for the first
                        #10_000 training iterations. This helps in spotting problems early.
                        if 0 <= counter <= 1 or not counter % save_result_every_n_batches or \
                           (0 <= counter < 10_000 and not counter % 1000) or counter == counter_init:

                            #Don't train on validation examples
                            if not val_counter % val_skip_n:
                                results = sess.run( monitoring_ops + output_ops, feed_dict=feed_dict )
                            else:
                                results = sess.run( monitoring_ops + output_ops + train_ops, feed_dict=feed_dict )

                            monitored_stats = results[:len(monitoring_ops)]
                            output = results[len(monitoring_ops)]

                            #Save images. Use PIL Image save feature as most basic libraries don't support the saving
                            #of signed float32 images
                            try:
                                save_input_loc = model_dir+"input-"+str(counter)+".tif"
                                save_truth_loc = model_dir+"truth-"+str(counter)+".tif"
                                save_output_loc = model_dir+"output-"+str(counter)+".tif"
                                Image.fromarray(_example_input[0].reshape(example_size[0], example_size[1]).astype(
                                    np.float32)).save( save_input_loc )
                                Image.fromarray(_example_output[0].reshape(example_size[0], example_size[1]).astype(
                                    np.float32)).save( save_truth_loc )
                                Image.fromarray(output.reshape(example_size[0], example_size[1]).astype(
                                    np.float32)).save( save_output_loc )
                            except:
                                print("Image save failed")

                        #For iterations where outputs don't need to be saved to disk, perform monitoring
                        #and training operations as usual
                        else:
                            #Don't train on validation examples
                            if not val_counter % val_skip_n:
                                results = sess.run( monitoring_ops, feed_dict=feed_dict )
                            else:
                                results = sess.run( monitoring_ops + train_ops, feed_dict=feed_dict )

                            monitored_stats = results[:len(monitoring_ops)]

                        #Prepare message for live performance update and to save to log file
                        my_network_name = "Fluffles-1"
                        message = f"{my_network_name}, Iter: {counter}, Val: {validating}"
                        for name, val in zip(monitoring_ops_names, monitored_stats):
                            message += f", {name}: {val}"

                        print(message)

                        message += "\n"
                        try:
                            log.write(message)
                        except:
                            print("Write to log failed")

                #Final save at the end of training
                saver.save(sess, save_path=model_dir, global_step=counter)
                quit() #Ends script

    return 

if __name__ == "__main__":
    main()