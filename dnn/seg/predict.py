# your implementation goes here

import numpy as np
from skimage import io

from train import FCN
from experiments.utils import predict_complete


def predict(image_name, weights_file, patch_size=(256, 256, 3), step=(220, 220, 3), threshold=0.5,
            out_file='images/out.png', zero_mean=True):
    """
    Predicts the output response map for an input image.
    :param image_name: Target image for prediction.
    :param weights_file: Path of model weights.
    :param patch_size: Window size.
    :param step: Stride length while predicting via sliding window.
    """
    image_data = io.imread(image_name)
    image_data = image_data[:, :, :3].astype(np.float32)
    if zero_mean:
        image_data = image_data - np.mean(image_data, axis=(0, 1))

    fcn_model = FCN()
    fcn_model.load_weights(weights_file)

    response_map = predict_complete(fcn_model, image_data, patch_size, step)
    out_image = np.zeros(response_map.shape[:2])
    out_image[np.where(response_map[:, :, 1] > threshold)] = 255

    io.imsave(out_file, out_image.astype(np.uint8))

    # plt.figure(1), plt.imshow(np.argmax(response_map, axis=-1))
    # plt.figure(2), plt.imshow(image_data.astype(np.uint32))
    # plt.show()
    #


if __name__ == '__main__':
    # predict('images/rgb.png', 'weights/model_exp1_4500', out_file='images/out1.png', zero_mean=False)
    predict('images/rgb.png', 'weights/model_exp3_4500', out_file='images/out3.png')
