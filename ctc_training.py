import tensorflow as tf
import argparse
import logging
import json
import os

from primus import CTC_PriMuS
import ctc_utils
import ctc_model

SAVE_PERIOD = 1000
IMG_HEIGHT = 128
MAX_EPOCHS = 64000
DROPOUT = 0.5
CONFIG_PATH = os.path.join(os.getcwd(),'config.json')

if tf.test.is_gpu_available():
    print("GPU is available")
else:
    print("GPU is not available")

# Calculate the sample error rate (SER) of the model
def validate(primus, params, sess, inputs, seq_len, rnn_keep_prob, decoded):
    validation_batch, validation_size = primus.getValidation(params)
    
    val_idx = 0
    
    val_ed = 0
    val_len = 0
    val_count = 0
        
    while val_idx < validation_size:
        mini_batch_feed_dict = {
            inputs: validation_batch['inputs'][val_idx:val_idx+params['batch_size']],
            seq_len: validation_batch['seq_lengths'][val_idx:val_idx+params['batch_size']],
            rnn_keep_prob: 1.0            
        }            
                    
        
        prediction = sess.run(decoded,
                            mini_batch_feed_dict)
    
        str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
    

        for i in range(len(str_predictions)):
            ed = ctc_utils.edit_distance(str_predictions[i], validation_batch['targets'][val_idx+i])
            val_ed = val_ed + ed
            val_len = val_len + len(validation_batch['targets'][val_idx+i])
            val_count = val_count + 1
            
        val_idx = val_idx + params['batch_size']
        
    return val_ed, val_len, val_count
    
# Train the model with the given parameters and save it to model_path
def train(corpus_path, set_path, voc_path, voc_type, model_path, validate_batches=False, verbose=False):   
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    assert os.path.exists(os.path.dirname(model_path)), 'Directory does not exist: ' + os.path.dirname(model_path)
    for path in [corpus_path, set_path, voc_path]: assert os.path.exists(path), 'File does not exist: ' + path

    # Set up tensorflow 
    tf.compat.v1.disable_eager_execution()        
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.InteractiveSession(config=config)

    # Load primus
    primus = CTC_PriMuS(corpus_path,set_path,voc_path, voc_type, val_split = 0.1)

    # Model parameters
    params = ctc_model.default_model_params(IMG_HEIGHT,primus.vocabulary_size)

    # Model
    inputs, seq_len, targets, decoded, loss, rnn_keep_prob = ctc_model.ctc_crnn(params)
    train_opt = tf.compat.v1.train.AdamOptimizer().minimize(loss)

    saver = tf.compat.v1.train.Saver(max_to_keep=None)
    sess.run(tf.compat.v1.global_variables_initializer())

    # Training loop
    for epoch in range(MAX_EPOCHS):
        
        batch = primus.nextBatch(params)
        _, loss_value = sess.run([train_opt, loss],
                                feed_dict={
                                    inputs: batch['inputs'],
                                    seq_len: batch['seq_lengths'],
                                    targets: ctc_utils.sparse_tuple_from(batch['targets']),
                                    rnn_keep_prob: DROPOUT,
                                }) 

        if epoch % SAVE_PERIOD == 0:
            # Validate
            logging.info('Loss value at epoch ' + str(epoch) + ':' + str(loss_value))
            
            if validate_batches:
                val_ed, val_len, val_count = validate(primus, params, sess, inputs, seq_len, rnn_keep_prob, decoded)
                logging.info('[Epoch ' + str(epoch) + '] ' + str(1. * val_ed / val_count) + ' (' + str(100. * val_ed / val_len) + ' SER) from ' + str(val_count) + ' samples.')    
            
            # Save model
            mdl_path = model_path + "-" +str(epoch)
            saver.save(sess,mdl_path,global_step=epoch)
            logging.info('Model saved to ' + mdl_path)

if __name__ == '__main__':
    configured_defaults = json.load(open(CONFIG_PATH))

    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('-corpus', dest='corpus', type=str, required=False, help='Path to the corpus.', default=configured_defaults['corpus_path'])
    parser.add_argument('-set',  dest='set', type=str, required=False, help='Path to the set file.', default=configured_defaults['set_path'])
    parser.add_argument('-save_model', dest='save_model', type=str, required=False, help='Path to save the model.', default=configured_defaults['model_path'])
    parser.add_argument('-vocabulary', dest='voc', type=str, required=False, help='Path to the vocabulary file.', default=configured_defaults['voc_path'])
    parser.add_argument('-voc_type', dest='voc_type', required=False, default=configured_defaults['voc_type'], choices=['semantic','agnostic'], help='Vocabulary type.')
    parser.add_argument('-validate_batches', dest='validate_batches', action="store_true", default=False, required=False)
    parser.add_argument('-verbose', dest='verbose', action="store_true", default=False, required=False)
    args = parser.parse_args()

    train(args.corpus, args.set, args.voc, args.voc_type, args.save_model, args.validate_batches, args.verbose)