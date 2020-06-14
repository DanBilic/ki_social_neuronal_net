from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet', include_top=False)

# Folder with images
img_folder = '/home/vkudinova/Desktop/dataset/cat/'

img_vector_features = []
names_indices = {}

j = 0
for file in sorted(os.listdir(img_folder)):
    filename = os.fsdecode(file)
    names_indices.update({j : filename})
    img_path = os.path.join(img_folder,filename)
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)
    vgg16_feature = np.array(vgg16_feature)
    vgg16_feature = vgg16_feature.flatten()
    img_vector_features.append(vgg16_feature)
    j += 1

# Query image
query_path = '/home/vkudinova/Desktop/redcat.jpg'

img = image.load_img(query_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg16_feature = model.predict(img_data)
vgg16_feature = np.array(vgg16_feature)
query_feature = vgg16_feature.flatten()

# Numbers of similar images that we want to show
N_QUERY_RESULT = 3
nbrs = NearestNeighbors(n_neighbors=N_QUERY_RESULT, metric="cosine").fit(img_vector_features)


distances, indices = nbrs.kneighbors([query_feature])
similar_image_indices = indices.reshape(-1)

best_fit = similar_image_indices[0]
filename = names_indices[best_fit]

print("The most similar image is", filename)

in_path = os.path.join(img_folder,filename)
img = image.load_img(in_path)
plt.imshow(img)
plt.show()




