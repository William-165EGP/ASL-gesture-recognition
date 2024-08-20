import tensorflow as tf

# This code is to test whether cuda is successfully installed
# Only one core is detected under M processor Mac
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
