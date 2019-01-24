#reference :https://blog.csdn.net/haima1998/article/details/80297710

from tensorflow.python import pywrap_tensorflow
import os
checkpoint_path = os.path.join( "model/model.ckpt-10350")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
