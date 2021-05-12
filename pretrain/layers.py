import tensorflow as tf

########################################
####### TRIPLET LOSS AS A LAYER ########
########################################

# functions for the triplet loss
def mask_valid_triplets(labels):
    """ returns 1 for valid triplets, 0 for invalid"""
    # i != j != k, so that anchor, positive, and negetives are distinct 
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    
    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)
    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)
    return tf.cast(mask, tf.float32)

def semi_hard_triplet_mining_mask(anchor_positive_d, anchor_negetive_d, margin):
    """ mask for d(a,p) <= d(a,n) <= d(a,p) + margin, returns 1 when true else 0 
        anchor_positive_d of shape (batch_size, batch_size, 1)
        anchor_negetive_d of shape (batch_size, 1, batch_size) """
    d_a_p_lessthan_d_a_n = tf.less_equal(anchor_positive_d, anchor_negetive_d)
    d_a_n_lessthan_d_a_p_plus_margin = tf.less_equal(anchor_negetive_d, anchor_positive_d + margin)
    mask = tf.logical_and(d_a_p_lessthan_d_a_n, d_a_n_lessthan_d_a_p_plus_margin)
    return tf.cast(mask, tf.float32)

def _get_anchor_positive_triplet_mask(labels):
    """mask[a, p] = 1 if a and p are distinct and have same label."""
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_and(indices_not_equal, labels_equal)
    return tf.cast(mask, tf.float32)

def _get_anchor_negative_triplet_mask(labels):
    """mask[a, n] = 1 if a and n have distinct labels"""
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_not(labels_equal)
    return tf.cast(mask, tf.float32)

# triplet loss as a layer
class TripletLoss(tf.keras.layers.Layer):
    def __init__(self, margin, strategy="all"):
        super(TripletLoss, self).__init__()
        self.margin = tf.cast(margin, tf.float32)
        self.epsilon = tf.keras.backend.epsilon()
        self.stratgy = strategy

    def call(self, embeddings, labels):
        if self.stratgy == "all":
            triplet_loss = self.batch_all_triplet_loss(embeddings, labels)
        else:    
            triplet_loss = self.batch_hard_triplet_loss(embeddings, labels)

        self.add_loss(triplet_loss)
        return embeddings
    
    def pairwise_euclidian_distances(self, embeddings):
        """ embeddings of shape (batch_size, depth) 
            returns squared euclidian distances"""
        # ||a-b||^2 = ||a||^2 - 2<a,b> + ||b||^2
        a_dot_b = tf.matmul(embeddings, embeddings, transpose_b=True)
        x_dot_x = tf.linalg.diag_part(a_dot_b)
        squared_distances = tf.expand_dims(x_dot_x, -1) - 2.0 * a_dot_b + tf.expand_dims(x_dot_x, 0)
        squared_distances = tf.maximum(squared_distances, 0.0) # for numerical stablity
        return squared_distances

    def batch_all_triplet_loss(self, embeddings, labels):
        distances = self.pairwise_euclidian_distances(embeddings)
        anchor_positive_d = tf.expand_dims(distances, 2)
        anchor_negetive_d = tf.expand_dims(distances, 1)
        temp_loss = triplet_loss = anchor_positive_d - anchor_negetive_d + self.margin

        valid_triplet_mask = mask_valid_triplets(labels)
        triplet_loss = tf.multiply(valid_triplet_mask, temp_loss)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # all_batch approach
        num_positive_losses = tf.reduce_sum(tf.cast(tf.greater(triplet_loss, self.epsilon), tf.float32))
        num_valid_triplets = tf.reduce_sum(valid_triplet_mask)
        fraction_positive_losses = num_positive_losses/num_valid_triplets
        triplet_loss = tf.reduce_sum(triplet_loss)/(num_positive_losses + self.epsilon)

        self.add_metric(fraction_positive_losses, name="fraction_positive_losses", aggregation='mean')
        return triplet_loss

    def batch_hard_triplet_loss(self, embeddings, labels):
        pairwise_dist = self.pairwise_euclidian_distances(embeddings)

        mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
        
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

        tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + self.margin, 0.0)
        triplet_loss = tf.reduce_mean(triplet_loss)
        self.add_metric(hardest_positive_dist, name="hardest_positive_dist", aggregation='mean')
        self.add_metric(hardest_negative_dist, name="hardest_negative_dist", aggregation='mean')
        return triplet_loss

# wraper function to incorporate triplet loss
def wrapper_tripletloss(model, margin, dist="cosine"):
    """ attaches triplet loss below the model """
    input_feats  = tf.keras.layers.Input(shape=model.input_shape[1:], name="feats")
    labels = tf.keras.layers.Input(shape=(None,), name="labels")
    embeddings = model(input_feats)
    embeddings = TripletLoss(margin)(embeddings, labels)
    training_model = tf.keras.Model([input_feats, labels], embeddings)
    return training_model

# mining_mask = semi_hard_triplet_mining_mask(anchor_positive_d, anchor_negetive_d, self.margin)
