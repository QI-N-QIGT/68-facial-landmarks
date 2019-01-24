# import tensorflow as tf
# with tf.Session() as sess:
#     with open('model/pb/frozen_model.pb', 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         print(graph_def)


import tensorflow as tf
with tf.Session() as sess:
    with open('model/pb/frozen_model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        print(graph_def)
        output = tf.import_graph_def(graph_def, return_elements=['Reshape/shape:output:0'])
        print(sess.run(output))
