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

SCALE_FACTOR_MIN = 0.4
SCALE_FACTOR_MAX = 1.5

ROT_ANGLE_MIN = -45
ROT_ANGLE_MAX = 45

BLUR_FACTOR_MIN = 0
BLUR_FACTOR_MAX = 12

CONTRAST_FACTOR_MIN = 0.0
CONTRAST_FACTOR_MAX = 1.0

BRIGHTNESS_FACTOR_MIN = 0.0
BRIGHTNESS_FACTOR_MAX = 1.0

SHARPEN_FACTOR_MIN = 0.0
SHARPEN_FACTOR_MAX = 1.0

TRANSLATION_OFFSET_MIN = -0.2
TRANSLATION_OFFSET_MAX = 0.2

SALT_PEPPER_FACTOR_MIN = 0.0
SALT_PEPPER_FACTOR_MAX = 0.0001

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

        return ctc_utils.rotate(image, angle)
    
class strech(augmentation):
    def __init__(self, variance, axis, distrobution = np.random.normal):
        super().__init__('strech', distrobution, variance)
        self.axis = int(axis)

    def augment(self, image):
        strech_factor = self.distrobution(1.0, self.variance)
        strech_factor = np.clip(strech_factor, SCALE_FACTOR_MIN, SCALE_FACTOR_MAX)

        im = ctc_utils.strech(image, strech_factor, self.axis)

        return im

class scale(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('scale', distrobution, variance)

    def augment(self, image):
        scale_factor = self.distrobution(1.0, self.variance)
        scale_factor = np.clip(scale_factor, SCALE_FACTOR_MIN, SCALE_FACTOR_MAX)

        return ctc_utils.scale(image, scale_factor)
    
class translate(augmentation):
    def __init__(self, variance, axis, distrobution = np.random.normal):
        super().__init__('translate', distrobution, variance)
        self.axis = int(axis)

    def augment(self, image):
        translate_factor = self.distrobution(0, self.variance)
        return ctc_utils.translate(image, translate_factor, self.axis)
    
class blur(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('blur', distrobution, variance)

    def augment(self, image):
        blur_factor = self.distrobution(0, self.variance)
        return ctc_utils.blur(image, blur_factor)

class contrast_shift(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('contrast_shift', distrobution, variance)

    def augment(self, image):
        contrast_factor = self.distrobution(0, self.variance)
        return ctc_utils.contrast_shift(image, contrast_factor)
    
class brightness_shift(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('brightness_shift', distrobution, variance)

    def augment(self, image):
        brightness_factor = self.distrobution(0, self.variance)
        return ctc_utils.brightness_shift(image, brightness_factor)
    
class sharpen(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('sharpen', distrobution, variance)

    def augment(self, image):
        sharpen_factor = self.distrobution(0, self.variance)
        return ctc_utils.sharpen(image, sharpen_factor)
    
class salt_pepper(augmentation):
    def __init__(self, variance, distrobution = np.random.normal):
        super().__init__('salt_pepper', distrobution, variance)

    def augment(self, image):
        salt_pepper_factor = self.distrobution(0, self.variance)
        np.clip(salt_pepper_factor, SALT_PEPPER_FACTOR_MIN, SALT_PEPPER_FACTOR_MAX)

        salt_pepper_mat = np.zeros(image.shape, dtype="uint8")
        if self.distrobution == np.random.normal:
            salt_pepper_mat = cv2.randn(salt_pepper_mat,0,self.variance)
            salt_pepper_mat = cv2.add(image, salt_pepper_mat)
        elif self.distrobution == np.random.uniform: #variance doesn't matter for uniform
            salt_pepper_mat = cv2.randu(salt_pepper_mat,0,255)
            salt_pepper_mat = salt_pepper_mat < self.variance
            salt_pepper_mat = salt_pepper_mat.astype(np.uint8)
            salt_pepper_mat = salt_pepper_mat * 255
            salt_pepper_mat = cv2.add(image, salt_pepper_mat)
        else:
            raise Exception('Invalid distrobution type \"' + str(self.distrobution) + '\"')       

        return salt_pepper_mat
    
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
        image = ctc_utils.scale(image, 0.5)
        # Define the transformation matrix (m)
        m = np.identity(3)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    m[i][j] = 1
                elif i == j:
                    m[i][j] = 1 + self.distrobution(0, self.variance)
                else:
                    m[i][j] = self.distrobution(0, self.variance)

        # Define the size of the output image (dsize)
        output_size = (image.shape[1], image.shape[0])  # Keep the same size as the input
        
        # Do not allow the image to be distorted out of frame
        m[0][2] = np.clip(m[0][2], -image.shape[1], image.shape[1])
        m[1][2] = np.clip(m[1][2], -image.shape[0], image.shape[0])

        # Apply the transformation matrix to the image
        output_image = cv2.warpPerspective(image, m, output_size, borderValue=255)

        # Crop the image to eliminate whike space
        output_image = ctc_utils.crop(output_image)

        return output_image
    