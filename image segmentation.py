import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

im=cv2.imread("Img01.jpg")
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(im)
plt.show()
reshaped_im = im.reshape((-1, 3))
print(reshaped_im.shape)

k = 2
kmeans = KMeans(k)

kmeans.fit(reshaped_im)

dominant_colors = kmeans.cluster_centers_.astype('uint8')

np.zeros_like(reshaped_im).shape
print(dominant_colors.shape, '*'*10)


#prominent colour extraction
plt.figure()
for i,color in enumerate(dominant_colors):
    palette = np.zeros_like(im, dtype='uint8')
    palette[:,:,:] = color
    plt.subplot(1,k,i+1)
    plt.axis("off")	
    plt.imshow(palette)
print(im.shape)

new_img = np.zeros((im.shape[0]*im.shape[1],3),dtype='uint8')
print(new_img.shape)
plt.figure()


#IMG seg
for ix in range (new_img.shape[0]):
    new_img[ix] = dominant_colors[kmeans.labels_[ix]]
    
new_img = new_img.reshape((im.shape))
plt.imshow(new_img)
plt.show()
