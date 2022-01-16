import cv2
from pywt import dwt2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("/home/andrejkronovsek/dfgc/Celeb-DF-v2-crop/Celeb-real/id0_0007/30.png")
wavename1 = 'haar'
wavename2 = 'bior1.3'
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (1024, 1024))

# Convert to float for more resolution for use with pywt
image = np.float32(image)
image /= 255

cA,(cH, cV, cD) = dwt2(image, wavename1)
cAA, (cAH, cAV, cAD) = dwt2(cA, wavename1)

#------ Izpis slike, ki vsebuje vseh 7 slik -------
Level2_1 = np.vstack([cAA, cAH])
Level2_2 = np.vstack([cAV, cAD])
Level2 = np.hstack([Level2_1, Level2_2])
Level1_bottom = np.vstack([Level2, cH])
Level1_right = np.vstack([cV, cD])
picture = np.hstack([Level1_bottom, Level1_right])
img = cv2.convertScaleAbs(picture, alpha=(255.0))
#------ Izpis slike, ki vsebuje vseh 7 slik -------

multi_channel_pic = []

for ix, i in enumerate([cAA, cAH, cAV, cAD, cH, cV, cD]):
    i = cv2.convertScaleAbs(i, alpha=(255.0/i.max()))
    i = cv2.resize(i, (299, 299), interpolation = cv2.INTER_CUBIC)
    #i = cv2.applyColorMap(i, cv2.COLORMAP_PINK)
    multi_channel_pic.append(i)
    cv2.imwrite('slika_cv2_' + str(ix) + '.png', i)

multi_channel_pic = np.array(multi_channel_pic)
multi_channel_pic = multi_channel_pic.flatten()
multi_channel_pic = multi_channel_pic.reshape(7, 299, 299)

print(multi_channel_pic.shape)
cv2.imwrite('slika_cv2.png', img)