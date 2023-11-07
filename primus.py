import cv2
import numpy as np
import ctc_utils
import ctc_otfa
import logging
import random

class CTC_PriMuS:
    gt_element_separator = '-'
    PAD_COLUMN = 0
    validation_dict = None

    FOLD_COEFFICIENT = 1 / 15000


    def __init__(self, corpus_dirpath, corpus_filepath, dictionary_path, voc_type, val_split = 0.0, distortion_ratio = 0.0, use_otfa=False):
        self.voc_type = voc_type
        self.corpus_dirpath = corpus_dirpath

        # Corpus
        corpus_file = open(corpus_filepath,'r')
        corpus_list = corpus_file.read().splitlines()
        corpus_file.close()

        self.fold_idx = -1
        self.distortion_phase = 0 # Offset for the distorted images
        self.distortion_ratio = distortion_ratio
        self.use_otfa = use_otfa

        if self.use_otfa:
            self.augmentations = ctc_otfa.read_augmentations()

        # Dictionary
        self.word2int = {}
        self.int2word = {}
            
        dict_file = open(dictionary_path,'r')
        dict_list = dict_file.read().splitlines()
        for word in dict_list:
            if not word in self.word2int:
                word_idx = len(self.word2int)
                self.word2int[word] = word_idx
                self.int2word[word_idx] = word

        dict_file.close()

        self.vocabulary_size = len(self.word2int)
        
        
        # Train and validation split
        random.shuffle(corpus_list) 
        val_idx = int(len(corpus_list) * val_split) 
        self.training_list = corpus_list[val_idx:]
        self.validation_list = corpus_list[:val_idx]

        # Split the training set into folds to account for limmited main-memory
        samples = len(self.training_list)
        self.folds = int(np.ceil(samples * self.FOLD_COEFFICIENT))
        self.fold_size = samples // self.folds
        
        
        logging.info ('Training with ' + str(len(self.training_list)) + ' and validating with ' + str(len(self.validation_list)))

    def read_images(self, params, fold_idx):
        images = []
        labels = []

        # This fold is already in buffer
        if self.fold_idx == fold_idx:
            return
 
        self.images = None
        self.labels = None
        
        # Read the fold into the buffer
        for sample in range(self.fold_size):
            sample_filepath = self.training_list[sample + fold_idx * self.fold_size]
            sample_fullpath = self.corpus_dirpath + '/' + sample_filepath + '/' + sample_filepath

            if self.distortion_ratio > 0:
                use_distorted = ((sample + self.distortion_phase)% (int(1 / self.distortion_ratio)) == 0)
            else:
                use_distorted = False

            try:
                if use_distorted:
                    if self.use_otfa:
                        sample_img = cv2.imread(sample_fullpath + '.png', cv2.IMREAD_GRAYSCALE)
                        sample_img = ctc_otfa.apply_augmentations(sample_img, self.augmentations)
                    else:
                        sample_img = cv2.imread(sample_fullpath + '_distorted.jpg', cv2.IMREAD_GRAYSCALE)
                else:
                    sample_img = cv2.imread(sample_fullpath + '.png', cv2.IMREAD_GRAYSCALE)
                assert len(sample_img.shape) == 2
            except FileNotFoundError:
                logging.warning('Image file not found: ' + sample_fullpath + ' (skipping sample)')
                continue

            height = params['img_height']
            sample_img = ctc_utils.resize(sample_img,height)
            images.append(ctc_utils.normalize(sample_img))

            sample_full_filepath = sample_fullpath + '.' + self.voc_type
            try:
                sample_gt_file = open(sample_full_filepath, 'r')
            except FileNotFoundError:
                logging.warning('Semantic file not found: ' + sample_full_filepath + ' (skipping sample)')
                continue
            
            sample_gt_plain = sample_gt_file.readline().rstrip().split(ctc_utils.word_separator())
            sample_gt_file.close()

            labels.append([self.word2int[lab] for lab in sample_gt_plain])

        self.images = images
        self.labels = labels
        self.fold_idx = fold_idx


    def get_batch(self, params,fold_idx, batch_idx):
        if fold_idx > self.folds:
            raise Exception('Invalid fold index')

        self.readImages(params,fold_idx)

        batch_start_idx = batch_idx * params['batch_size']
        batch_end_idx = (batch_idx + 1) * params['batch_size']
        if batch_end_idx > self.fold_size:
            batch_end_idx = self.fold_size
        
        images = self.images[batch_start_idx:batch_end_idx]
        labels = self.labels[batch_start_idx:batch_end_idx]

        # Transform to batch
        image_widths = [img.shape[1] for img in images]
        max_image_width = max(image_widths)

        batch_images = np.ones(shape=[params['batch_size'],
                                       params['img_height'],
                                       max_image_width,
                                       params['img_channels']], dtype=np.float32)*self.PAD_COLUMN

        for i, img in enumerate(images):
            batch_images[i, 0:img.shape[0], 0:img.shape[1], 0] = img

        # LENGTH
        width_reduction = 1
        for i in range(params['conv_blocks']):
            width_reduction = width_reduction * params['conv_pooling_size'][i][1]

        lengths = [ batch_images.shape[2] / width_reduction ] * batch_images.shape[0]

        return {
            'inputs': batch_images,
            'seq_lengths': np.asarray(lengths),
            'targets': labels,
        }
        
    def getValidation(self, params):
        if self.validation_dict == None:                
            images = []
            labels = []
    
            # Read files
            for sample_filepath in self.validation_list:
                sample_fullpath = self.corpus_dirpath + '/' + sample_filepath + '/' + sample_filepath
                sample_img = cv2.imread(sample_fullpath + '.png', cv2.IMREAD_GRAYSCALE)
                assert len(sample_img.shape) == 2
                height = params['img_height']
                sample_img = ctc_utils.resize(sample_img,height)
                images.append(ctc_utils.normalize(sample_img))
    
                sample_full_filepath = sample_filepath + '.' + self.voc_type            
                sample_gt_file = open(sample_full_filepath, 'r')
            
                sample_gt_plain = sample_gt_file.readline().rstrip().split(ctc_utils.word_separator())
                sample_gt_file.close()
    
                labels.append([self.word2int[lab] for lab in sample_gt_plain])
    
            # Transform to batch
            image_widths = [img.shape[1] for img in images]
            max_image_width = max(image_widths)
    
            batch_images = np.ones(shape=[len(self.validation_list),
                                           params['img_height'],
                                           max_image_width,
                                           params['img_channels']], dtype=np.float32)*self.PAD_COLUMN
    
            for i, img in enumerate(images):
                batch_images[i, 0:img.shape[0], 0:img.shape[1], 0] = img
    
            # LENGTH
            width_reduction = 1
            for i in range(params['conv_blocks']):
                width_reduction = width_reduction * params['conv_pooling_size'][i][1]
    
            lengths = [ batch_images.shape[2] / width_reduction ] * batch_images.shape[0]
    
            self.validation_dict = {
                'inputs': batch_images,
                'seq_lengths': np.asarray(lengths),
                'targets': labels,
            }            
        
        return self.validation_dict, len(self.validation_list)
