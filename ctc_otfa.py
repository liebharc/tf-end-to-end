import numpy as np
import argparse
import logging
import cv2
import ctc_utils
import os 
import json
from abc import ABC, abstractmethod
import ctc_otfa

AUGS_PATH = os.path.join(os.getcwd(),'otfa.json')

def read_augmentations(augmentations_path=AUGS_PATH):
    # read from the file path and return a list of augmentation objects
    with open(augmentations_path) as json_file:
        data = json.load(json_file)
        augmentations = []
        
        # use the 'type' field literally to create the augmentation object
        for aug in data:
            if aug['type'] == 'rotation':
                augmentations.append(ctc_otfa.rotation(aug['variance'],aug['distrobution']))
            elif aug['type'] == 'strech':
                augmentations.append(ctc_otfa.strech(aug['variance'], aug['axis'],aug['distrobution']))
            elif aug['type'] == 'scale':
                augmentations.append(ctc_otfa.scale(aug['variance'],aug['distrobution']))
            elif aug['type'] == 'translate':
                augmentations.append(ctc_otfa.translate(aug['variance'], aug['axis'],aug['distrobution']))
            elif aug['type'] == 'blur':
                augmentations.append(ctc_otfa.blur(aug['variance'],aug['distrobution']))
            elif aug['type'] == 'contrast_shift':
                augmentations.append(ctc_otfa.contrast_shift(aug['variance'],aug['distrobution']))
            elif aug['type'] == 'brightness_shift':
                augmentations.append(ctc_otfa.brightness_shift(aug['variance'],aug['distrobution']))
            elif aug['type'] == 'sharpen':
                augmentations.append(ctc_otfa.sharpen(aug['variance'],aug['distrobution']))
            elif aug['type'] == 'salt_pepper':
                augmentations.append(ctc_otfa.salt_pepper(aug['variance'],aug['distrobution']))
            else:
                raise Exception('Invalid augmentation type \"' + aug['type'] + '\"')
            
        return augmentations
    
def apply_augmentations(image, augmentations):
    for aug in augmentations:
        if aug.variance > 0.0:
            image = aug.augment(image)
    return image

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--augmentations-path", required=False, default=AUGS_PATH, help="Path to augmentations json")
    ap.add_argument("-i", "--image-path", required=True, help="Path to image")
    ap.add_argument("-o", "--output-path", required=False, help="Path to output image")
    args = vars(ap.parse_args())

    augmentations = read_augmentations()
    for aug in augmentations:
        print(str(aug) + ' '), 

    image = cv2.imread(args["image_path"], cv2.IMREAD_GRAYSCALE)
    image = apply_augmentations(image, augmentations)

    if args["output_path"] is not None:
        cv2.imwrite(args["output_path"], image)
    else:
        cv2.imshow("Augmented", image)
        cv2.waitKey(0)

class augmentation(ABC):
    def __init__(self, type, distrobution, variance):
        self.type = str(type)
        self.distrobution = distrobution
        self.variance = float(variance)

    def __str__(self) -> str:
        return 'AUGMENTATION: ' +self.type + ', ' + str(self.variance) + ', ' + str(self.distrobution)

    @abstractmethod
    def augment():
        pass

class rotation(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('rotation', np.random.normal, variance)

    def augment(self, image):
        angle = self.distrobution(0, self.variance)
        return ctc_utils.rotate(image, angle)
    
class strech(augmentation):
    def __init__(self, variance, axis, distrobution = np.random.normal):
        super().__init__('strech', np.random.normal, variance)
        self.axis = axis

    def augment(self, image):
        strech_factor = self.distrobution(0, self.variance)
        return ctc_utils.strech(image, strech_factor)

class scale(augmentation):
    def __init__(self, variance, axis, distrobution = np.random.normal):
        super().__init__('scale', np.random.normal, variance)
        self.axis = axis

    def augment(self, image):
        scale_factor = self.distrobution(0, self.variance)
        return ctc_utils.scale(image, scale_factor)
    
class translate(augmentation):
    def __init__(self, variance, axis, distrobution = np.random.normal):
        super().__init__('translate', np.random.normal, variance)
        self.axis = axis

    def augment(self, image):
        translate_factor = self.distrobution(0, self.variance)
        return ctc_utils.translate(image, translate_factor)
    
class blur(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('blur', np.random.normal, variance)

    def augment(self, image):
        blur_factor = self.distrobution(0, self.variance)
        return ctc_utils.blur(image, blur_factor)

class contrast_shift(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('contrast_shift', np.random.normal, variance)

    def augment(self, image):
        contrast_factor = self.distrobution(0, self.variance)
        return ctc_utils.contrast_shift(image, contrast_factor)
    
class brightness_shift(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('brightness_shift', np.random.normal, variance)

    def augment(self, image):
        brightness_factor = self.distrobution(0, self.variance)
        return ctc_utils.brightness_shift(image, brightness_factor)
    
class sharpen(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('sharpen', np.random.normal, variance)

    def augment(self, image):
        sharpen_factor = self.distrobution(0, self.variance)
        return ctc_utils.sharpen(image, sharpen_factor)
    
class salt_pepper(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('salt_pepper', np.random.normal, variance)

    def augment(self, image):
        salt_pepper_factor = self.distrobution(0, self.variance)
        return ctc_utils.salt_pepper(image, salt_pepper_factor)