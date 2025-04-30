import numpy as np
import gzip
import struct
from scipy.ndimage import rotate, shift, zoom

def augment_image(img, mode='rotate', param=10):
    """
    Perform basic augmentation on a single 28x28 image.
    :param img: numpy array, shape (28, 28)
    :param mode: 'rotate', 'shift', or 'zoom'
    :param param: parameter for augmentation (angle in degrees, shift in pixels, zoom factor)
    :return: augmented image
    """
    if mode == 'rotate':
        # Rotate the image around center, with reshape=False to keep original size
        return rotate(img, angle=param, reshape=False, mode='nearest')
    elif mode == 'shift':
        # Shift image in x and y directions
        return shift(img, shift=[param, param], mode='nearest')
    elif mode == 'zoom':
        # Zoom in or out the image
        zoomed = zoom(img, zoom=param)
        if param < 1.0:
            # Pad zoomed image to 28x28
            pad = (28 - zoomed.shape[0]) // 2
            return np.pad(zoomed, ((pad, 28 - zoomed.shape[0] - pad), (pad, 28 - zoomed.shape[1] - pad)), mode='constant')
        else:
            # Crop zoomed image to 28x28
            crop = (zoomed.shape[0] - 28) // 2
            return zoomed[crop:crop+28, crop:crop+28]
    else:
        raise ValueError("Unsupported augmentation mode.")

def augment_dataset(images, labels, num_augmented=4):
    """
    Augment each image with transformations to increase dataset size.
    :param images: numpy array of shape (N, 28, 28)
    :param labels: numpy array of shape (N,)
    :param num_augmented: number of augmented versions per original image
    :return: augmented_images, augmented_labels
    """
    augmented_images = []
    augmented_labels = []
    for img, label in zip(images, labels):
        augmented_images.append(img)
        augmented_labels.append(label)
        for _ in range(num_augmented):
            mode = np.random.choice(['rotate', 'shift', 'zoom'])
            param = {
                'rotate': np.random.uniform(-15, 15),
                'shift': np.random.uniform(-2, 2),
                'zoom': np.random.uniform(0.9, 1.1)
            }[mode]
            aug_img = augment_image(img, mode, param)
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)

def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape((num, rows, cols)).astype(np.float32) / 255.0  # Normalize to [0,1]
    return images

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

import numpy as np
import gzip
import struct
import os

# 保存为 IDX 格式的图像文件
def save_idx_images(images, filepath):
    # images: numpy array of shape (num_images, 28, 28)
    num_images = images.shape[0]
    rows, cols = images.shape[1], images.shape[2]
    with gzip.open(filepath, 'wb') as f:
        # magic number for images is 2051 (0x00000803)
        f.write(struct.pack('>IIII', 2051, num_images, rows, cols))
        f.write((images * 255).astype(np.uint8).tobytes())  # scale back to [0,255]

# 保存为 IDX 格式的标签文件
def save_idx_labels(labels, filepath):
    # labels: numpy array of shape (num_images,)
    num_labels = labels.shape[0]
    with gzip.open(filepath, 'wb') as f:
        # magic number for labels is 2049 (0x00000801)
        f.write(struct.pack('>II', 2049, num_labels))
        f.write(labels.astype(np.uint8).tobytes())

# Load training data
train_images_path = r'dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'dataset/MNIST/train-labels-idx1-ubyte.gz'

images = load_mnist_images(train_images_path)
labels = load_mnist_labels(train_labels_path)

print("Loaded images:", images.shape)
print("Loaded labels:", labels.shape)

# 使用数据增强函数
aug_images, aug_labels = augment_dataset(images, labels, num_augmented=4)

print("After augmentation:", aug_images.shape, aug_labels.shape)

# 合并图片与标签
combined_images = np.concatenate([images, aug_images], axis=0)
combined_labels = np.concatenate([labels, aug_labels], axis=0)

# 检查
print("Combined images:", combined_images.shape)
print("Combined labels:", combined_labels.shape)

# 保存文件路径
out_images_path = 'dataset/MNIST/train-images-augmented-idx3-ubyte.gz'
out_labels_path = 'dataset/MNIST/train-labels-augmented-idx1-ubyte.gz'

# 写入压缩文件
save_idx_images(combined_images, out_images_path)
save_idx_labels(combined_labels, out_labels_path)

print("Saved augmented MNIST files.")

