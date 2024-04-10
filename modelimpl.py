import cv2
from cv2 import dnn_superres
import numpy as np


sr = dnn_superres.DnnSuperResImpl_create()

path = 'EDSR_x4.pb'
sr.readModel(path)
sr.setModel('edsr', 4)


image = cv2.imread('Blurry-low-quality-female-portrait-picture.jpg')


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


denoised = cv2.fastNlMeansDenoisingColored(image, None, h=5, hColor=5, templateWindowSize=6, searchWindowSize=21)


upscale = sr.upsample(denoised)


cv2.imwrite('upscaled.png', upscale)


bicubic = cv2.resize(denoised, (upscale.shape[1], upscale.shape[0]), interpolation=cv2.INTER_CUBIC)

cv2.imwrite('denoised.png', bicubic)


kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  
sharpened = cv2.filter2D(bicubic, -1, kernel_sharpening)


cv2.imwrite('sharpened.png', sharpened)
