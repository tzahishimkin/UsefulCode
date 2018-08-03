from __future__ import absolute_import, division, print_function

import tensorflow as tf


'''
1. tansor.numpy() - gets value of tensor
2. Do not put variable inside tf network
3, inherent from tf.keras.Model (Or tf.keras.Network)
4. Check out tf.contrib. what is it
5. model.variables - returns all models's variables
6. save doesn't work. only model.save_weights
7. The input dimentions are built when calling to call. Not prior to it
8. when building a model, inherit from tf.keras.Modle. 
    you need to set suepr.__init__() in init function
    you need to put model in call 
9. check out tf.contrib.checkpoint.list - read about it. This should be useful
10. tf.GradienTape() 
tape.gradient(loss, variables) tp get gradoemts pf tje ;pss 

11. 
 gradients = tape.gradient(loss, self.variables)  # Calcuate gradients 
self.optimizer.apply_gradients(zip(gradients, self.variables)) # Apply gradients to weights
return loss

12. with tf.device('/gpu:0') read if this is still needed. He said that from TF1.8 this is not needed

13. Read when you fo @static in python

14. building custom layer:
__init__ - Prepare eveything without input shape
build - recieves input shape. Need to call super.build
call -  adding the operations and activations

15. data pipelie - Runs in a graph structure on the CPU:
def _normalize(x):
    x = x / 255.0
    x -= mean_val
    return x

train_ds = tf.data.Dataset.from_tensor_slices(x_train).map(_normalize, num_parallel_calls=4)  #from_generator only needs to support inter. Nothing else (as 
train_ds = train_ds.apply(tf.contrib.data.shuffle_and_repeat(len(x_train), num_epochs))
train_ds = train_ds.batch(batch_size).apply(tf.contrib.data.prefetch_to_device("/gpu:0")) #prepare batch in GPU

Another example:
def _add_length(x, y):
    x = x[:max_len]
    x_dict = {"seq": x, "seq_len": tf.size(x)}
    return x_dict, y

train_ds = tf.data.Dataset.from_generator(lambda: zip(x_train, y_train), output_types=(tf.int32, tf.int32),
                                          output_shapes=([None], []))
train_ds = train_ds.map(_add_length, num_parallel_calls=4)
train_ds = train_ds.apply(tf.contrib.data.shuffle_and_repeat(len(x_train), num_epochs))
train_ds = train_ds.padded_batch(batch_size, padded_shapes=({"seq": [None], "seq_len": []}, []))
train_ds = train_ds.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))

16. Add regulizer:
loss+=tf.reduc_sum(self.loss)

17. check out what tf.variables / eager->variables are 

18. what replace the name scole from TF

19. 

Link:
https://colab.research.google.com/drive/1yiknVxL9rvwlU4XAYXJXwzjTGfDoUqh2#scrollTo=0H6oVqxf9KgK
'''
tf.enable_eager_execution()


tf.keras.Model

tf.executing_eagerly()        # => True

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))


a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
# => tf.Tensor([[1 2]
#               [3 4]], shape=(2, 2), dtype=int32)

# Broadcasting support
b = tf.add(a, 1)
print(b)
# => tf.Tensor([[2 3]
#               [4 5]], shape=(2, 2), dtype=int32)

# Operator overloading is supported
print(a * b)
# => tf.Tensor([[ 2  6]
#               [12 20]], shape=(2, 2), dtype=int32)

# Use NumPy values
import numpy as np

c = np.multiply(a, b)
print(c)
# => [[ 2  6]
#     [12 20]]

# Obtain numpy value from a tensor:
print(a.numpy())
# => [[1 2]
#     [3 4]]


tfe = tf.contrib.eager


def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(max_num.numpy()):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num)
    counter += 1
  return counter



class MySimpleLayer(tf.keras.layers.Layer):
  def __init__(self, output_units):
    super(MySimpleLayer, self).__init__()
    self.output_units = output_units

  def build(self, input_shape):
    # The build method gets called the first time your layer is used.
    # Creating variables on build() allows you to make their shape depend
    # on the input shape and hence removes the need for the user to specify
    # full shapes. It is possible to create variables during __init__() if
    # you already know their full shapes.
    self.kernel = self.add_variable(
      "kernel", [input_shape[-1], self.output_units])

  def call(self, input):
    # Override call() instead of __call__ so we can perform some bookkeeping.
    return tf.matmul(input, self.kernel)
