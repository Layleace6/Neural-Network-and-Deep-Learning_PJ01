import numpy as np

class Layer:
    """
    Base class for all layers.
    """
    def __init__(self):
        self.training = True
        self.optimizable = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

def he_init(size):
    """
    He initialization for weights.
    Args:
        size (tuple): Shape of the weight matrix.
    """
    fan_in = np.prod(size[1:]) if len(size) > 2 else size[0]
    return np.random.randn(*size) * np.sqrt(2. / fan_in)

class Linear(Layer):
    """
    Fully connected linear layer.
    """
    def __init__(self, in_dim, out_dim, initialize_method=he_init, weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.W = initialize_method((in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        if X.ndim >2:
            X = X.reshape(X.shape[0], -1)
        output = np.dot(X, self.W) + self.b
        return output

    def backward(self, grad_output):
        batch_size = self.input.shape[0]
        self.grads['W'] = np.dot(self.input.T, grad_output) / batch_size
        self.grads['b'] = np.sum(grad_output, axis=0, keepdims=True) / batch_size

        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        grad_input = np.dot(grad_output, self.W.T)
        return grad_input

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class conv2D(Layer):
    """
    A 2D Convolution Layer without automatic padding by default.
    Input shape: [batch_size, C_in, H_in, W_in]
    Output shape: [batch_size, C_out, H_out, W_out]
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=np.random.normal,
                 weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self.W = initialize_method(size=(out_channels, in_channels, *self.kernel_size))
        self.b = np.zeros((out_channels, 1))

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.optimizable = True

        self.input = None  # Cached input
        self.X_col = None  # Cached im2col input
        self.X_padded = None  # Cached padded input

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        Forward pass of conv2D.
        """
        batch_size, C_in, H_in, W_in = X.shape
        K, _, kH, kW = self.W.shape

        # Calculate output dimensions
        H_out = (H_in + 2 * self.padding - kH) // self.stride + 1
        W_out = (W_in + 2 * self.padding - kW) // self.stride + 1

        # Apply zero-padding if necessary
        if self.padding > 0:
            X_padded = np.pad(X, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        else:
            X_padded = X

        # Save for backward
        if self.training:
            self.X_padded = X_padded

        # Efficient im2col: unfold all patches at once
        X_col = np.lib.stride_tricks.as_strided(
            X_padded,
            shape=(batch_size, C_in, H_out, W_out, kH, kW),
            strides=(
                X_padded.strides[0],
                X_padded.strides[1],
                self.stride * X_padded.strides[2],
                self.stride * X_padded.strides[3],
                X_padded.strides[2],
                X_padded.strides[3],
            ),
            writeable=False,
        )
        X_col = X_col.reshape(batch_size, C_in * kH * kW, H_out * W_out)

        if self.training:
            self.X_col = X_col

        # Weight matrix reshape
        W_col = self.W.reshape(K, -1)

        # Matrix multiplication
        out = np.einsum('kc,bcp->bkp', W_col, X_col) + self.b.reshape(1, K, 1)
        out = out.reshape(batch_size, K, H_out, W_out)

        return out

    def backward(self, grad_output):
        """
        Backward pass of conv2D.
        """
        batch_size, K, H_out, W_out = grad_output.shape
        _, C_in, kH, kW = self.W.shape

        # Reshape grad_output to [batch_size, K, H_out * W_out]
        grad_output_flat = grad_output.reshape(batch_size, K, -1)

        # Weight matrix reshape
        W_col = self.W.reshape(K, -1)

        # Compute gradients w.r.t. weights and bias
        dW = np.einsum('bkp,bcp->kc', grad_output_flat, self.X_col) / batch_size
        dW = dW.reshape(self.W.shape)
        db = np.sum(grad_output_flat, axis=(0,2)).reshape(self.b.shape) / batch_size

        # Compute gradients w.r.t. input
        dX_col = np.einsum('kc,bkp->bcp', W_col, grad_output_flat)

        # Reshape dX_col back to image
        dX_padded = np.zeros_like(self.X_padded)
        for idx in range(H_out * W_out):
            i = idx // W_out
            j = idx % W_out
            h_start = i * self.stride
            w_start = j * self.stride
            patch = dX_col[:, :, idx].reshape(batch_size, C_in, kH, kW)
            dX_padded[:, :, h_start:h_start + kH, w_start:w_start + kW] += patch

        # Remove padding
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        # Weight decay (L2 regularization)
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        # Save gradients
        self.grads['W'] = dW
        self.grads['b'] = db

        return dX

    def clear_grad(self):
        """
        Clear stored gradients.
        """
        self.grads = {'W': None, 'b': None}


class ReLU(Layer):
    """
    ReLU activation function.
    """
    def __init__(self):
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return np.maximum(0, X)

    def backward(self, grad_output):
        grad = grad_output.copy()
        grad[self.input <= 0] = 0
        return grad


class MultiCrossEntropyLoss(Layer):
    """
    Multi-class cross-entropy loss with optional softmax.
    """
    def __init__(self, model=None, max_classes=10):
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.preds = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.labels = labels
        if self.has_softmax:
            exps = np.exp(predicts - np.max(predicts, axis=1, keepdims=True))
            self.preds = exps / np.sum(exps, axis=1, keepdims=True)
        else:
            self.preds = predicts

        batch_size = predicts.shape[0]
        epsilon = 1e-12
        loss = -np.mean(np.log(self.preds[np.arange(batch_size), labels] + epsilon))
        return loss

    def backward(self):
        batch_size = self.preds.shape[0]
        self.grads = self.preds.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


class L2Regularization:
    """
    L2 Regularization.
    Given a model (or layers), applies L2 penalty on weights.
    """

    def __init__(self, model, weight_decay_lambda=1e-4):
        """
        Args:
            model: the entire model or a list of layers having 'params' dict.
            weight_decay_lambda: the strength of L2 regularization.
        """
        self.model = model
        self.lambda_ = weight_decay_lambda

    def loss(self):
        """
        Compute the L2 regularization loss term: (lambda / 2) * sum(W^2)
        """
        l2_loss = 0.0
        layers = self.model if isinstance(self.model, list) else [self.model]

        for layer in layers:
            if hasattr(layer, 'params'):
                for name, param in layer.params.items():
                    if name == 'W':  # Only penalize weights, not biases
                        l2_loss += np.sum(param ** 2)

        return 0.5 * self.lambda_ * l2_loss

    def backward(self):
        """
        Add L2 regularization gradients into layer grads.
        Each W gets an additional grad term: lambda * W
        """
        layers = self.model if isinstance(self.model, list) else [self.model]

        for layer in layers:
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for name, param in layer.params.items():
                    if name == 'W':
                        if layer.grads[name] is not None:
                            layer.grads[name] += self.lambda_ * param
                        else:
                            layer.grads[name] = self.lambda_ * param

    def __call__(self):
        """
        Make the class callable to directly compute the L2 regularization loss.
        """
        return self.loss()


class MaxPool2D(Layer):
    """
    A 2D Max Pooling Layer implementation.
    """
    def __init__(self, kernel_size=2, stride=2):
        """
        Initialize the MaxPool2D layer.
        
        Args:
            kernel_size (int): Size of the pooling window (default: 2).
            stride (int): Stride of the pooling operation (default: 2).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None          # Stores the input during forward pass
        self.argmax_mask = None    # Stores the mask of max indices for backpropagation
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        Perform the forward pass of max pooling.
        
        Args:
            X (numpy.ndarray): Input tensor of shape [B, C, H, W].
        
        Returns:
            numpy.ndarray: Output tensor after max pooling of shape [B, C, H_out, W_out].
        """
        self.input = X  # Store input for backpropagation
        B, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride

        # Calculate output dimensions
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        # Extract sliding windows using strides
        shape = (B, C, H_out, W_out, k, k)
        strides = (
            X.strides[0],  # Batch dimension stride
            X.strides[1],  # Channel dimension stride
            X.strides[2] * s,  # Height dimension stride with stride applied
            X.strides[3] * s,  # Width dimension stride with stride applied
            X.strides[2],      # Kernel height stride
            X.strides[3],      # Kernel width stride
        )
        patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

        # Reshape patches to compute max values
        reshaped = patches.reshape(B, C, H_out, W_out, -1)
        out = reshaped.max(axis=-1)  # Compute max values across the kernel

        # Save argmax mask for backpropagation during training
        if self.training:
            max_mask = reshaped == out[..., None]  # Create a boolean mask of max positions
            self.argmax_mask = max_mask.reshape(patches.shape)

        return out

    def backward(self, grads):
        """
        Perform the backward pass of max pooling.
        
        Args:
            grads (numpy.ndarray): Gradient of the loss w.r.t. the output of this layer,
                                   of shape [B, C, H_out, W_out].
        
        Returns:
            numpy.ndarray: Gradient of the loss w.r.t. the input of this layer,
                           of shape [B, C, H, W].
        """
        B, C, H, W = self.input.shape
        k = self.kernel_size
        s = self.stride

        # Calculate output dimensions
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        # Initialize gradient of input
        dX = np.zeros_like(self.input)

        # Reshape argmax mask for broadcasting
        mask = self.argmax_mask.reshape(B, C, H_out, W_out, k * k)
        grads_expand = grads[..., None]  # Add an extra dimension for broadcasting

        # Apply gradients only to the max positions
        grads_broadcasted = grads_expand * mask
        grads_broadcasted = grads_broadcasted.reshape(B, C, H_out, W_out, k, k)

        # Accumulate gradients into the input gradient tensor
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * s
                w_start = j * s
                dX[:, :, h_start:h_start+k, w_start:w_start+k] += grads_broadcasted[:, :, i, j, :, :]

        return dX


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)
    

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


def he_init(size):
    """
    He initialization for weights.
    Args:
        size (tuple): Shape of the weight matrix.
    """
    fan_in = np.prod(size[1:]) if len(size) > 2 else size[0]
    return np.random.randn(*size) * np.sqrt(2. / fan_in)
