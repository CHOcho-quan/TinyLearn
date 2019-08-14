import numpy as np
import cv2
import pickle

class MyConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.

    """

    def __init__(self, input_dim=(3, 96, 96), num_filters=32, filter_size=5,
                 hidden_dim=100, num_classes=2, weight_scale=1e-3, reg=1.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(int((H / 2)*(W / 2)*num_filters), hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['gamma'] = np.zeros(hidden_dim)
        self.params['beta'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def fc_forward(self, x, w, b):
        """
        Computes the forward pass for an fc layer.

        """
        # print(x.shape, w.shape, b.shape)
        out = None

        x_n = x.reshape(x.shape[0], -1)
        out = x_n.dot(w) + b
        cache = (x, w, b)

        return out, cache

    def fc_backward(self, dout, cache):
        """
        Computes the backward pass for an fc layer.

        """
        x, w, b = cache
        dx, dw, db = None, None, None

        N = x.shape[0]
        # print(x.shape)
        x_rsp = x.reshape(N , -1)
        dx = dout.dot(w.T)
        dx = dx.reshape(*x.shape)
        dw = x_rsp.T.dot(dout)
        db = np.sum(dout, axis = 0)

        return dx, dw, db


    def relu_forward(self, x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        """
        out = None
        out = x * (x >= 0)
        cache = x
        return out, cache


    def relu_backward(self, dout, cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        """
        dx, x = None, cache
        dx = dout * (x >= 0)
        return dx

    def conv_forward(self, x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width HH.

        """
        out = None

        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        H_out = 1 + (H + 2 * pad - HH) / stride
        W_out = 1 + (W + 2 * pad - WW) / stride
        out = np.zeros((N , F , int(H_out), int(W_out)))

        x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        for i in range(int(H_out)):
            for j in range(int(W_out)):
                x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                for k in range(F):
                    out[:, k , i, j] = np.sum(x_pad_masked * w[k, :, :, :], axis=(1,2,3))

        out = out + (b)[None, :, None, None]

        cache = (x, w, b, conv_param)
        return out, cache

    def conv_backward(self, dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.

        """
        dx, dw, db = None, None, None
        x, w, b, conv_param = cache

        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        H_out = 1 + (H + 2 * pad - HH) / stride
        W_out = 1 + (W + 2 * pad - WW) / stride

        x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        dx = np.zeros_like(x, dtype=self.dtype)
        dx_pad = np.zeros_like(x_pad, dtype=self.dtype)
        dw = np.zeros_like(w, dtype=self.dtype)
        db = np.zeros_like(b, dtype=self.dtype)

        db = np.sum(dout, axis = (0,2,3))

        x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        for i in range(int(H_out)):
            for j in range(int(W_out)):
                x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                for k in range(F): #compute dw
                    dw[k ,: ,: ,:] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
                for n in range(N): #compute dx_pad
                    # print(dx_pad.dtype.name, dout.dtype.name, w.dtype.name)
                    dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum((w[:, :, :, :] *
                                                                                (dout[n, :, i, j])[:,None ,None, None]), axis=0)
        dx = dx_pad[:,:,pad:-pad,pad:-pad]
        return dx, dw, db

    def max_pool_forward(self, x, pool_param):
        """
        A naive implementation of the forward pass for a max pooling layer.

        """
        out = None

        N, C, H, W = x.shape
        HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        H_out = (H-HH)/stride+1
        W_out = (W-WW)/stride+1
        out = np.zeros((N,C,int(H_out),int(W_out)))
        for i in range(int(H_out)):
            for j in range(int(W_out)):
                x_masked = x[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW]
                out[:,:,i,j] = np.max(x_masked, axis=(2,3))
        cache = (x, pool_param)
        return out, cache

    def max_pool_backward(self, dout, cache):
        """
        A naive implementation of the backward pass for a max pooling layer.

        """
        dx = None

        x, pool_param = cache
        N, C, H, W = x.shape
        HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        H_out = (H-HH)/stride+1
        W_out = (W-WW)/stride+1
        dx = np.zeros_like(x)

        for i in range(int(H_out)):
            for j in range(int(W_out)):
                x_masked = x[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW]
                max_x_masked = np.max(x_masked,axis=(2,3))
                temp_binary_mask = (x_masked == (max_x_masked)[:,:,None,None])
                dx[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW] += temp_binary_mask * (dout[:,:,i,j])[:,:,None,None]

        return dx

    def conv_relu_pool_forward(self, x, w, b, conv_param, pool_param):
        """
        Convenience layer that performs a convolution, a ReLU, and a pool.

        """
        a, conv_cache = self.conv_forward(x, w, b, conv_param)
        s, relu_cache = self.relu_forward(a)
        out, pool_cache = self.max_pool_forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    def batchnorm_forward(self, x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.


        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
        running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

        out, cache = None, None
        if mode == 'train':
            sample_mean = np.mean(x, axis = 0)
            sample_var = np.var(x , axis = 0)
            x_hat = (x - sample_mean) / (np.sqrt(sample_var  + eps))
            out = gamma * x_hat + beta
            cache = (gamma, x, sample_mean, sample_var, eps, x_hat)
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
        elif mode == 'test':
            scale = gamma / (np.sqrt(running_var  + eps))
            out = x * scale + (beta - running_mean * scale)
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

        return out, cache

    def batchnorm_backward(self, dout, cache):
        """
        Backward pass for batch normalization.

        """
        dx, dgamma, dbeta = None, None, None
        gamma, x, u_b, sigma_squared_b, eps, x_hat = cache
        N = x.shape[0]

        dx_1 = gamma * dout
        dx_2_b = np.sum((x - u_b) * dx_1, axis=0)
        dx_2_a = ((sigma_squared_b + eps) ** -0.5) * dx_1
        dx_3_b = (-0.5) * ((sigma_squared_b + eps) ** -1.5) * dx_2_b
        dx_4_b = dx_3_b * 1
        dx_5_b = np.ones_like(x) / N * dx_4_b
        dx_6_b = 2 * (x - u_b) * dx_5_b
        dx_7_a = dx_6_b * 1 + dx_2_a * 1
        dx_7_b = dx_6_b * 1 + dx_2_a * 1
        dx_8_b = -1 * np.sum(dx_7_b, axis=0)
        dx_9_b = np.ones_like(x) / N * dx_8_b
        dx_10 = dx_9_b + dx_7_a

        dgamma = np.sum(x_hat * dout, axis=0)
        dbeta = np.sum(dout, axis=0)
        dx = dx_10

        return dx, dgamma, dbeta

    def conv_relu_pool_backward(self, dout, cache):
        """
        Backward pass for the conv-relu-pool convenience layer

        """
        conv_cache, relu_cache, pool_cache = cache
        ds = self.max_pool_backward(dout, pool_cache)
        da = self.relu_backward(ds, relu_cache)
        dx, dw, db = self.conv_backward(da, conv_cache)
        return dx, dw, db

    def softmax_loss(self, x, y):
        """
        Computes the loss and gradient for softmax classification.

        """
        # print(x.shape, y.shape)
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        print("probs", probs)
        N = x.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        conv_forward_out_1, cache_forward_1 = self.conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
        fc_forward_out_2, cache_forward_2 = self.fc_forward(conv_forward_out_1, self.params['W2'], self.params['b2'])
        fc_relu_2, cache_relu_2 = self.relu_forward(fc_forward_out_2)
        bn_3, cache_bn_3 = self.batchnorm_forward(fc_relu_2, self.params['gamma'], self.params['beta'], {'mode':'train'})
        scores, cache_forward_3 = self.fc_forward(bn_3, self.params['W3'], self.params['b3'])

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dout = self.softmax_loss(scores, y)

        # Add regularization
        loss += self.reg * 0.5 * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2))

        dX4, grads['W3'], grads['b3'] = self.fc_backward(dout, cache_forward_3)
        dX3, grads['gamma'], grads['beta'] = self.batchnorm_backward(dX4, cache_bn_3)
        dX2 = self.relu_backward(dX3, cache_relu_2)
        dX2, grads['W2'], grads['b2'] = self.fc_backward(dX2, cache_forward_2)
        dX1, grads['W1'], grads['b1'] = self.conv_relu_pool_backward(dX2, cache_forward_1)

        grads['W3'] = grads['W3'] + self.reg * self.params['W3']
        grads['W2'] = grads['W2'] + self.reg * self.params['W2']
        grads['W1'] = grads['W1'] + self.reg * self.params['W1']

        return loss, grads

if __name__ == '__main__':
    myCNN = MyConvNet()
