import numpy as np
from PIL import Image

from siamese import Siamese

if __name__ == "__main__":
    model = Siamese()

    image_1 = Image.open('img/Angelic_01.png')

    image_2 = Image.open('img/Angelic_02.png')

    probability = model.detect_image(image_1, image_2)
    print(probability)
