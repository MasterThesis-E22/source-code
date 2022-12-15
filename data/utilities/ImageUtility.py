import numpy as np
from PIL import Image as Image


class ImageUtility:

    @staticmethod
    def load_embryos_image(path, focal, frame, width, height):
        file_data = np.load(path)
        images = file_data['images']

        focal = focal
        frame = frame
        img_raw = images[frame, :, :, focal]
        img = Image.fromarray(img_raw)
        img = ImageUtility.resize_image(img, width, height)
        return img

    @staticmethod
    def resize_image(image, width, height):
            newSize = (width, height)
            img = image.resize(newSize)
            return img

    @staticmethod
    def normalize_image(image):
            img_raw = np.asarray(image)
            img_raw = img_raw.astype('float32') / 255
            return img_raw