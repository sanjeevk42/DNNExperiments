# your implementation goes here
import os

import numpy as np
import torch
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from experiments.utils import SemSegDataset, predict_complete

import tensorflow as tf


class FCN(Module):
    """
    Fully convolutional module which outputs logits on forward pass.
    """

    def __init__(self, in_channels=3, n_classes=2):
        super(FCN, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        self.logits = nn.Conv2d(in_channels=16, out_channels=self.n_classes, kernel_size=5, padding=2)

        self.softmax = nn.Softmax2d()

    def forward(self, inp):
        x = self.relu1(self.conv1(inp))
        x = self.maxpool(self.relu2(self.conv2(x)))
        x = self.upsample(self.relu3(self.conv3(x)))
        x = self.logits(x)
        return x

    def predict_patch(self, img_patch):
        """
        Predicts the output response map for given patch.
        This function assumes -1 dim as channel in input and output.
        :param img_patch: Numpy image array of shape compatible with network.
        :return: Output response map.
        """
        input_tensor = torch.from_numpy(np.transpose(np.expand_dims(img_patch, axis=0), (0, 3, 1, 2)))
        prob_out = self.softmax(self(input_tensor))
        prob_out = np.transpose(np.squeeze(prob_out.data.numpy(), 0), (1, 2, 0))
        return prob_out

    def load_weights(self, weight_file):
        self.load_state_dict(torch.load(weight_file))


def save_weights(weights_dir, model_state, step):
    os.makedirs(weights_dir, exist_ok=True)
    weight_file = os.path.join(weights_dir, 'model_{}'.format(step))
    torch.save(model_state, weight_file)


def as_summary(key, value):
    return tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])


if __name__ == '__main__':

    epochs = 5000
    lr = 1e-3
    batch_size = 1
    class_balance = False

    if class_balance:
        weights = torch.from_numpy(np.array([1., 11], dtype=np.float32))
    else:
        weights = torch.from_numpy(np.array([1., 1], dtype=np.float32))

    model = FCN()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = SemSegDataset(patch_sel='random')
    data_loader = DataLoader(dataset, batch_size=batch_size)
    summary_writer = tf.summary.FileWriter('training_logs')

    itr = 0
    for i in range(epochs):
        for batch in data_loader:
            image_batch = batch['image']
            gt_batch = batch['gt']

            y_pred = model(image_batch)
            loss = loss_fn(y_pred, gt_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            summary_writer.add_summary(as_summary('loss', loss.item()), itr)
            itr += 1

        print('Step:{}, loss:{}'.format(i, loss.item()))

        summary_writer.flush()

        if i % 500 == 0 or i + 1 == epochs:
            response_map = predict_complete(model, dataset.image_data)
            plt.figure(1), plt.imshow(dataset.image_data)
            plt.figure(2), plt.imshow(response_map[:, :, 0])
            plt.show()
            save_weights('weights', model.state_dict(), i)
