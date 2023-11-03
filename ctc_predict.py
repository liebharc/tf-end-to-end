import argparse
import tensorflow as tf
import ctc_utils
import cv2
import numpy as np

# Predict the semantic string of a music score image
def predict(image_path, model_path, voc_file):
    # Disable eager execution
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.InteractiveSession()

    # Read the vocabulary to convert the integer predictions to words
    dict_file = open(voc_file,'r')
    dict_list = dict_file.read().splitlines()
    int2word = dict()
    for word in dict_list:
        word_idx = len(int2word)
        int2word[word_idx] = word
    dict_file.close()

    # Restore weights
    saver = tf.compat.v1.train.import_meta_graph(model_path)
    saver.restore(sess,model_path[:-5])

    graph = tf.compat.v1.get_default_graph()

    input = graph.get_tensor_by_name("model_input:0")
    seq_len = graph.get_tensor_by_name("seq_lengths:0")
    rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
    height_tensor = graph.get_tensor_by_name("input_height:0")
    width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
    logits = tf.compat.v1.get_collection("logits")[0]

    # Constants that are saved inside the model itself
    WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

    decoded, _ = tf.compat.v1.nn.ctc_greedy_decoder(logits, seq_len)

    image = cv2.imread(image_path)
    #convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = ctc_utils.resize(image, HEIGHT)
    image = ctc_utils.normalize(image)
    image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

    seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

    prediction = sess.run(decoded,
                        feed_dict={
                            input: image,
                            seq_len: seq_lengths,
                            rnn_keep_prob: 1.0,
                        })

    str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
    semantic_string = ''
    for w in str_predictions[0]:
        semantic_string += (int2word[w]) +'\t'
    semantic_string = semantic_string[:-1]
    
    return semantic_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
    parser.add_argument('-image',  dest='image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('-vocabulary', dest='vo c_file', type=str, required=True, help='Path to the vocabulary file.')
    args = parser.parse_args()
    
    print(predict(args.image, args.model, args.voc_file))