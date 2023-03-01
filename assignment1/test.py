import numpy as np 
from task2a import pre_process_images
import utils
import matplotlib.pyplot as plt


X_train, Y_train, X_val, Y_val = utils.load_full_mnist()


img = X_train[206].reshape(28,28) 

plt.imsave("test.png", img, cmap="gray")

plt.imshow(img)
plt.show()




