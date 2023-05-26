import cv2
import random
import albumentations as A


def dilate(img):
    if random.randint(1, 5) == 1:  # p=0.2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        img = cv2.dilate(img, kernel, iterations=1)
    return img


def erode(img):
    if random.randint(1, 5) == 1:  # p=0.2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        img = cv2.erode(img, kernel, iterations=1)
    return img


def black_pixel_noise(p=0.5):
    transform = A.OneOf([
        A.RandomRain(brightness_coefficient=1.0,
                     drop_length=2,
                     drop_width=2,
                     drop_color=(0,0,0),
                     blur_value=1,
                     rain_type='drizzle',
                     p=0.05),
        A.RandomShadow(p=1),
    ], p=p)
    
    return transform


def white_pixel_noise(p=0.5):
    transform = A.OneOf([
        A.RandomRain(brightness_coefficient=1.0,
                     drop_length=2,
                     drop_width=2,
                     drop_color=(255,255,255),
                     blur_value=1,
                     rain_type=None,
                     p=1),
    ], p=p)

    return transform


def color_jitter(p=0.5):
    transform = A.ColorJitter(brightness=0.2,
                              contrast=0.2,
                              saturation=0.2,
                              hue=0.2,
                              always_apply=False,
                              p=p)
    return transform


def blur(p=0.5):
    transform = A.Blur(blur_limit=5, p=p)
    return transform
