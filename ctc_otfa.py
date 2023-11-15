import numpy as np
import argparse
import logging
import cv2
import ctc_utils
import os 
import json
from abc import ABC, abstractmethod
import ctc_otfa
import tensorflow as tf

AUGS_PATH = os.path.join(os.getcwd(),'otfa.json')

SCALE_FACTOR_MIN = 0.75
SCALE_FACTOR_MAX = 1.5

ROT_ANGLE_MIN = -45
ROT_ANGLE_MAX = 45

BLUR_FACTOR_MIN = 0
BLUR_FACTOR_MAX = 12

CONTRAST_FACTOR_MIN = 0.0
CONTRAST_FACTOR_MAX = 1.0

BRIGHTNESS_DELTA_MIN = -1.0
BRIGHTNESS_DELTA_MAX = 1.0

SHARPEN_FACTOR_MIN = 0.0
SHARPEN_FACTOR_MAX = 1.0

TRANSLATION_OFFSET_MIN = -0.2
TRANSLATION_OFFSET_MAX = 0.2

SALT_PEPPER_FACTOR_MIN = 0.0
SALT_PEPPER_FACTOR_MAX = 1.0

RADIAL_DISTORTION_FACTOR_MAX = 0.000001
RADIAL_DISTORTION_FACTOR_MIN = -RADIAL_DISTORTION_FACTOR_MAX

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
            elif aug['type'] == 'radial_distortion':
                augmentations.append(ctc_otfa.radial_distortion(aug['variance'],aug['distrobution']))
            elif aug['type'] == 'distortion':
                augmentations.append(ctc_otfa.distort(aug['variance'],aug['distrobution']))
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
    ap.add_argument("-a", "--augmentations-path", dest="augmentations_path", required=False, default=AUGS_PATH, help="Path to augmentations json")
    ap.add_argument("-i", "--image-path", dest="image_path", required=True, help="Path to image")
    ap.add_argument("-o", "--output-path", dest="output_path", required=False, help="Path to output image")
    args = ap.parse_args()

    while True:
        augmentations = read_augmentations(args.augmentations_path)
        for aug in augmentations:
            print(str(aug) + ' '), 

        image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
        image = apply_augmentations(image, augmentations)

        if args.output_path is not None:
            cv2.imwrite(args.output_path, image)
            break
        else:
            cv2.imshow("Augmented", image)
            cv2.waitKey(0)


class augmentation(ABC):
    def __init__(self, type, distrobution, variance):
        self.type = str(type)
        
        if distrobution == 'normal' or distrobution == 'gaussian':
            self.distrobution = np.random.normal
        elif distrobution == 'uniform':
            self.distrobution = np.random.uniform
        elif isinstance(distrobution, np.random.Generator):
            self.distrobution = distrobution
        else:
            raise Exception('Invalid distrobution type \"' + str(distrobution) + '\"')
        self.variance = float(variance)

    def __str__(self) -> str:
        return 'AUGMENTATION: ' +self.type + ', ' + str(self.variance) + ', ' + str(self.distrobution)

    # This function should take a tensor and return a tensor
    @abstractmethod
    def augment():
        pass

class rotation(augmentation):
    def __init__(self, variance, distrobution = np.random.normal, min=0.0):
        super().__init__('rotation', distrobution, variance)

    def augment(self, image):
        angle = self.distrobution(0, self.variance)
        angle = np.clip(angle, ROT_ANGLE_MIN, ROT_ANGLE_MAX)
        angle = angle + 1
        
        #convert to tensor
        #return tf.numpy_function(ctc_utils.rotate, [image, angle], tf.float32)
    
        pass 
class strech(augmentation):
    def __init__(self, variance, axis, distrobution = np.random.normal):
        super().__init__('strech', distrobution, variance)
        self.axis = int(axis)

    def augment(self, image_tensor):
        pass

class scale(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('scale', distrobution, variance)

    def augment(self, image):
        pass
    
class translate(augmentation):
    def __init__(self, variance, axis, distrobution = np.random.normal):
        super().__init__('translate', distrobution, variance)
        self.axis = int(axis)

    def augment(self, image):
        pass
    
class blur(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('blur', distrobution, variance)

    def augment(self, image):
        pass

class contrast_shift(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('contrast_shift', distrobution, variance)

    def augment(self, image):
        if self.distrobution == np.random.normal:
            lower = 1.0 - np.abs(self.distrobution(0, self.variance))
            upper = 1.0 + np.abs(self.distrobution(0, self.variance))
            lower = np.clip(lower, CONTRAST_FACTOR_MIN, upper)
            upper = np.clip(upper, lower, CONTRAST_FACTOR_MAX)
        elif self.distrobution == np.random.uniform:
            lower = 1.0 -  self.variance
            lower = np.clip(lower, CONTRAST_FACTOR_MIN, CONTRAST_FACTOR_MAX)
            upper = 1.0 + self.variance
            upper = np.clip(upper, CONTRAST_FACTOR_MIN, CONTRAST_FACTOR_MAX)
        else:
            raise Exception('Invalid distrobution type \"' + str(self.distrobution) + '\"')

        return tf.image.random_contrast(image, lower=lower, upper=upper)
    
class brightness_shift(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('brightness_shift', distrobution, variance)

    def augment(self, image):
        if self.distrobution == np.random.normal:
            brightness_delta = self.distrobution(0, self.variance)
        elif self.distrobution == np.random.uniform:
            brightness_delta = self.distrobution(-self.variance, self.variance)

        brightness_delta = np.clip(brightness_delta, BRIGHTNESS_DELTA_MIN, BRIGHTNESS_DELTA_MAX)
        return tf.image.adjust_brightness(image, delta=brightness_delta)
    
class sharpen(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('sharpen', distrobution, variance)

    def augment(self, image):
        pass
    
class salt_pepper(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('salt_pepper', distrobution, variance)

    def augment(self, image):
        if self.distrobution == np.random.normal:
            noise = tf.random.normal(tf.shape(image), 0, 1)
        elif self.distrobution == np.random.uniform:
            noise = tf.random.uniform(tf.shape(image), 0, 1)
        
        salt_noise = np.abs(self.distrobution(0, self.variance))
        
        salt_noise = np.clip(salt_noise, SALT_PEPPER_FACTOR_MIN, SALT_PEPPER_FACTOR_MAX)        
        image_with_salt = tf.minimum(image + salt_noise, 1.0)

        pepper_noise = tf.where(noise > (1.0 - image), 0.0, 1.0)
        image_with_salt_and_pepper = tf.maximum(image_with_salt - pepper_noise, 0.0)

        return image_with_salt_and_pepper
    
class radial_distortion(augmentation):
    def __init__(self, variance, distrobution):
        super().__init__("radial_distortion", distrobution, variance)

    def augment(self, image):
        k1 = self.distrobution(0, self.variance)
        k1 = np.clip(k1, RADIAL_DISTORTION_FACTOR_MIN, RADIAL_DISTORTION_FACTOR_MAX)
        k2 = self.distrobution(0, self.variance)
        k2 = np.clip(k2, RADIAL_DISTORTION_FACTOR_MIN, RADIAL_DISTORTION_FACTOR_MAX)

        return ctc_utils.radial_distortion(image, k1, k2)

class distort(augmentation):
    def __init__(self, variance, distrobution):
        super().__init__("distortion", distrobution, variance)
    
    def augment(self, image):
        pass
