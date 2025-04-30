# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

def MLP(train_imgs, valid_imgs):
        #  normalize from [0, 255] to [0, 1]
        train_imgs = train_imgs.astype(np.float32) / 255.
        valid_imgs = valid_imgs.astype(np.float32) / 255.
        linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
        optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
        loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

        runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
        model = runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')
        runner.visualize_first_layer_weights_MLP(model, save_path="figs/mlp_weights.png")
        return runner

def CNN(train_imgs, valid_imgs):
        # CNN model version:
        train_imgs = train_imgs.reshape(-1, 1, 28, 28).astype(np.float32) / 255.
        valid_imgs = valid_imgs.reshape(-1, 1, 28, 28).astype(np.float32) / 255.
        cnn_model = nn.models.Model_CNN()

        optimizer = nn.optimizer.SGD(init_lr=0.12, model=cnn_model)
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
        loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

        runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

        model = runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=8, log_iters=100, save_dir=r'./best_models')
        runner.visualize_first_layer_weights_CNN(model, save_path="figs/cnn_weights.png")
        return runner

runner = MLP(train_imgs, valid_imgs)
_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()