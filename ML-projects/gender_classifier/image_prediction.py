import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
model = tf.keras.models.load_model('gender_classify.h5')

def load_preprocess_image(path):
    """
    :param path: Path to the image that we will test
    :return: Return proccesd image and original image
    """
    img_org = image.load_img(path,target_size=(64,64))
    img = image.img_to_array(img_org)
    img = np.expand_dims(img,axis=0)
    print(img.shape)
    return img,img_org

def predict_image(path):
    """
    :param path:
    Predict the preproccesd image and plot the original image.
    """
    image,org = load_preprocess_image(path)
    image = np.vstack([image])
    class_of_image = model.predict(image, batch_size=1)
    print(class_of_image[0])
    if class_of_image[0] > 0.5:
        gender = "Male"
    else:
        gender = "Female"
    plt.imshow(org)
    plt.xlabel("Model predict this image as [" + gender + "] with accuracy : " + str(class_of_image[0]))
    plt.show()
path = "./cursed.png"
predict_image(path)