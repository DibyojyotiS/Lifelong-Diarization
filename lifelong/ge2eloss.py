import tensorflow as tf
import numpy as np

def ge2e(x, labels, centroids, w: tf.Variable, b: tf.Variable, eps=0.01):
    """
        x: network output of shape BxTxD B is the Batch size
        labels: array of corresponding labels shaped BxT
        centroids: list with B sublists containing class centroids
                   dimention of a sublist is KxD where K is #classes
                   in that sublist, K may vary for different sublists.
        w and b: tensorflow variables, dtype: tf.float32

        note 0: all dtypes are assumed to be tf.float32
        note 1: x and centroids are assumed to be l2-normalized
        note 2: subtracting an embedding vector 'v' from the 
                corresponding centroid is assumed to move it by 
                eps * v, giving: ckv = (1+eps)*ck - eps*v; eps<<1
        note 3: centroids should be ordered by label
        note 4: loss is weighted by inverse sqrt of #embd
    """
    B = len(labels)
    x = tf.cast(x, tf.float32)

    Lg = []
    for i in range(B):

        centroids_ = tf.convert_to_tensor(centroids[i], dtype=tf.float32) # shape KxD
        centroids_ = tf.expand_dims(centroids_, 1) # shape Kx1xD

        x_ = tf.slice(x, (i, 0, 0), (1,-1,-1)) # shape 1xTxD
        x_ = tf.tile(x_, [centroids_.get_shape()[0],1,1]) # shape KxTxD

        _, labels_ = tf.unique(labels[i]) # shape T,
        labels_mask = tf.one_hot(labels_, x_.get_shape()[0]) # shape TxK
        labels_mask = tf.transpose(labels_mask) # shape KxT
        num_embds = tf.reduce_sum(labels_mask, axis=1, keepdims=True) # shape Kx1

        ckik = tf.linalg.l2_normalize((1+eps)*centroids_ - eps*x_, axis=-1) # shape KxTxD

        cos_kik = tf.reduce_sum(tf.multiply(ckik, x_), axis=-1) # shape KxT
        cos_jik = tf.squeeze(tf.matmul(centroids_, x_, transpose_b=True)) # shape KxT

        sjik = labels_mask*cos_kik + (1.0 - labels_mask)*cos_jik # shape KxT
        sjik = w * sjik + b # shape KxT

        Leji = -tf.nn.log_softmax(sjik, axis=0) # shape KxT
        Leji = tf.divide(Leji, tf.sqrt(num_embds)) # shape KxT
        Leji = tf.boolean_mask(Leji, labels_mask) # shape T,
        Leji = tf.reduce_sum(Leji)

        Lg.append(Leji)

    Lg = tf.reduce_sum(Lg)/B
    return Lg
        


def vge2e(x, labels, centroids, w: tf.Variable, b: tf.Variable, eps=0.01):
    """
        fully vectorized version

        x: network output of shape BxTxD B is the Batch size
        labels: array of corresponding labels shaped BxT
        centroids: list with B sublists containing class centroids
                   dimention of a sublist is KxD where K is #classes
                   in that sublist, K may vary for different sublists.
        w and b: tensorflow variables, dtype: tf.float32

        note 0: all dtypes are assumed to be tf.float32
        note 1: x and centroids are assumed to be l2-normalized
        note 2: subtracting an embedding vector 'v' from the 
                corresponding centroid is assumed to move it by 
                eps * v, giving: ckv = (1+eps)*ck - eps*v; eps<<1
        note 3: centroids should be ordered by label
        note 4: there is no class weighting of the loss
    """

    x = tf.cast(x, tf.float32)

    centroids_ = tf.ragged.constant(centroids, dtype=tf.float32) 
    centroids_ = centroids_.to_tensor() # shape BxKxD
    centroids_ = tf.expand_dims(centroids_, 2) # shape BxKx1xD

    x_ = tf.expand_dims(x, axis=1) # shape Bx1xTxD
    x_ = tf.tile(x_, [1, centroids_.get_shape()[1],1,1]) # shape BxKxTxD

    labels_ = tf.map_fn(lambda x: tf.unique(x)[1], tf.convert_to_tensor(labels)) # shape BxT
    labels_mask = tf.one_hot(labels_, x_.get_shape()[1]) # shape BxTxK
    labels_mask = tf.transpose(labels_mask, perm=[0, 2, 1]) # shape BxKxT
    labels_present = tf.reduce_max(labels_mask, axis=-1, keepdims=True) # shape BxKx1

    ckik = tf.linalg.l2_normalize((1+eps)*centroids_ - eps*x_, axis=-1) # shape BxKxTxD

    cos_kik = tf.reduce_sum(tf.multiply(ckik, x_), axis=-1) # shape BxKxT
    cos_jik = tf.squeeze(tf.matmul(centroids_, x_, transpose_b=True)) # shape BxKxT

    sjik = labels_mask*cos_kik + (1.0 - labels_mask)*cos_jik # shape BxKxT
    sjik = w * sjik + b # shape BxKxT

    sjik = tf.where(sjik != 0, sjik, -np.inf)

    Leji = -tf.nn.log_softmax(sjik, axis=1) # shape BxKxT
    Leji = tf.boolean_mask(Leji, labels_mask) # shape B*T,
    
    Lg = tf.reduce_sum(Leji)/len(labels)
    return Lg


# # demo
# # labels like [0, 2, 2, 4, 1, 5, 8] - missing 3,6,7 are acceptable now
# # centroids must still be ordered by corresponding label magnitude
# 
# x = tf.cast(np.random.randn(16, 20, 5), tf.float32)
# labels = [
#       np.random.randint(0, np.random.randint(1,30), 20) for _ in range(16)
#     ]
# centroids = [
#     list(np.random.randn(len(np.unique(labels[i])), 5)) for i in range(16)
# ]
# w = tf.Variable(1.0)
# b = tf.Variable(0.0)
# eps = 0.1
# b1 = ge2e(x, labels, centroids, w, b, eps)
# b2 = vge2e(x, labels, centroids, w, b, eps)
# print(b1,b2, np.isclose(b1,b2))