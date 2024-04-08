
import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from IPython.display import display, Image

# Display original image
image_path = 'image.jpg'

# Read the image
image = plt.imread(image_path)

# Display the image
plt.imshow(image)
plt.show()


# Convert image to array, print out the shape of array, and print out the entire array
img_matrix = np.asarray(image)
print(img_matrix.shape)
print(img_matrix)



######## Convolution with Laplacian Filter ##################

lap_ker = np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]])


output_img = np.zeros_like(image, dtype=float)

#centroids = []



for i in range(image.shape[0] - lap_ker.shape[0] + 1):
    for j in range(image.shape[1] - lap_ker.shape[1] + 1):
        output_img[i, j] = np.sum(image[i:i+3, j:j+3] * lap_ker)



#print( output_img[0:5, 0:5])
print( output_img[:10])
print(output_img.shape)

plt.imshow(output_img)
plt.show()

"""

stride=2
Validation for first 5X5 array (from upper-left corner), i.e., filtered_results[0:5,0:5]. The example figure is below.

[[ 134.   98.  173.    5.    3.]\
 [  96. -163.   68.  -10.   37.]\
 [   7. -141. -127.  142.   -6.]\
 [  -1.  -46.  109.  -13.   11.]\
 [ 106.   49.  241.  -26.  -33.]]



"""

######## Convolution with Laplacian Filter and the setting of stride=2 ##################
output_img = np.zeros_like(image, dtype=float)
for i in range(1, image.shape[0] - 1, 2):
    for j in range(1, image.shape[1] - 1, 2):
        output_img[i//2, j//2] = np.sum(image[i-1:i+2, j-1:j+2] * lap_ker)

#print( output_img[0:5, 0:5])
print( output_img[:10])
print(output_img.shape)

plt.imshow(output_img)
plt.show()

"""
 Prepare a 2X2 pooling mask.
 Conduct max pooing on image with prepared mask.
Validation for first 5X5 array (from upper-left corner), i.e., pooled_results[0:5,0:5].The example figure is below.

[[ 98. 112.  93. 195. 173.]\
 [ 84. 127. 137. 253. 254.]\
 [ 85. 145. 225. 255. 242.]\
 [104. 178. 216. 230. 242.]\
 [ 95. 186. 147. 248. 242.]]

"""

######## MaxPooling with the setting of 2X2 ##################

pool_mask = np.zeros((2, 2))
pool_mask[0, 0] = 1
pool_mask[1, 1] = 1

pool_img = np.zeros((image.shape[0] // 2, image.shape[1] //2) )

for i in range(0, image.shape[0], 2):
    for j in range(0, image.shape[1], 2):
        window = image[i:i+2, j:j+2]
        max_value = np.max(window)
        pool_img[int(i/2), int(j/2)] = max_value

plt.imshow(pool_img)
plt.show()
