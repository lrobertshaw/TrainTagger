import tensorflow as tf

def custom_loss(y_true, y_pred, alpha=1, beta=1, epsilon=1e-3):
    err_rel = (y_pred - y_true) / (y_true + epsilon)
    mape = tf.math.reduce_mean( tf.math.abs(err_rel) )    # mean absolute percentage error
    sdpe = tf.math.reduce_std(err_rel)    # standard deviation of the relative error
    return (alpha * mape) + (beta * sdpe)    # loss is a weighted sum of MAPE and standard deviation of the relative error