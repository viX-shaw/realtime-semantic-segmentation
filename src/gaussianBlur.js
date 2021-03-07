import * as tf from '@tensorflow/tfjs';
// import { Tensor3D, Tensor2D, Tensor1D, Tensor4D, Tensor } from '@tensorflow/tfjs';


function get1dGaussianKernel(sigma, size){
    // Generate a 1d gaussian distribution across a range
    var x = tf.range(Math.floor(-size / 2) + 1, Math.floor(size / 2) + 1)
    x = tf.pow(x, 2)
    x = tf.exp(x.div(-2.0 * (sigma * sigma)))
    x = x.div(tf.sum(x))
    return x
}

function get2dGaussianKernel(size, sigma) {
    // This default is to mimic opencv2. 
    sigma = sigma || (0.3 * ((size - 1) * 0.5 - 1) + 0.8)

    var kerne1d = get1dGaussianKernel(sigma, size)
    return tf.outerProduct(kerne1d, kerne1d)
}

export function getAverageKernel(size) {
    var x = tf.abs(tf.range(Math.floor(-size / 2) + 1, Math.floor(size / 2) + 1))
    // var x = tf.fill([size], 0.02)
    // var x = tf.tensor1d([2,1,2])
    x = tf.outerProduct(x, x).divNoNan(tf.scalar(5))
    // x = tf.stack([x, x, x])
    // let x = tf.fill([size, size], 0.25)
    return tf.reshape(x, [size, size, 1, 1])
}

export function getGaussianKernel(size = 5, sigma) {
    return tf.tidy(() => {
        var kerne2d = get2dGaussianKernel(size, sigma)
        var kerne3d = tf.stack([kerne2d, kerne2d, kerne2d])
        return tf.reshape(kerne3d, [size, size, 3, 1])
    })
}

export function blur(image, kernel) {
    return tf.tidy(() => {
        return tf.depthwiseConv2d(image, kernel, 1, "valid")
    })
}