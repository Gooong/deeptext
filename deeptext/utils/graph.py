# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

import os

import tensorflow as tf

from tensorflow.python.framework import graph_util

FILENAME_FROZEN_GRAPH = "frozen_model.pb"

def freeze_graph(model_dir, output_node_names):
        # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
            
            # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = os.path.join(absolute_model_folder, FILENAME_FROZEN_GRAPH)

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
                            
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

        # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

            # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
                                sess, # The session is used to retrieve the weights
                                input_graph_def, # The graph_def is used to retrieve the nodes 
                                output_node_names
                            ) 

                    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


def load_graph(model_dir):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
    frozen_graph_filename = os.path.join(model_dir, FILENAME_FROZEN_GRAPH)
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph
