import tensorflow as tf
import numpy as np

def get_train_data():
    '''
    X_train.shape: (N, 2)
    y_train.shape: (N,)
    '''
    X_train = np.array([[1, 1], [1, 2]])
    y_train = np.array([1, 2])
    return X_train, y_train

def model(X):
    '''
    W.shape: (2, 1)
    b.shape: (1,)
    y.shape: (N, 1)
    return : (N,)
    '''
    W = tf.get_variable("W", shape=[2, 1])
    b = tf.get_variable("b", shape=[1])
    y = tf.matmul(X, W) + b
    return y[:, 0]

def run(variables, feed_dict, epoches):
    '''
    print loss, prediction, weights every 10 iterations
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # for o in sess.graph.get_operations():
        #     print(o.name)
        # for v in tf.global_variables():
        #     print(v.name)
        W = sess.graph.get_tensor_by_name("W:0")
        b = sess.graph.get_tensor_by_name("b:0")
        for e in range(epoches):
            pred_val, loss_val, _ = sess.run(variables, feed_dict = feed_dict)
            if e % 10 == 0:
                print("Epoch {0}, loss = {1:.3g}, ".format(e + 1, loss_val))
                print("prediction:", pred_val)
                print("W:\n", W.eval())
                print("b:\n", b.eval())

def main():
    # placeholders
    X = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [None])
    # graph
    pred = model(X)
    # loss
    loss = tf.sqrt(tf.losses.mean_squared_error(pred, y))
    # optimizer
    optimizer = tf.train.AdamOptimizer(0.1)
    training  = optimizer.minimize(loss)
    # training data
    X_train, y_train = get_train_data()
    print("X_train.shape:", X_train.shape, "y_train.shape:", y_train.shape)
    # run tensorflow
    variables = [pred, loss, training]
    feed_dict = {X: X_train, y: y_train}
    epoches = 100
    run(variables, feed_dict, epoches)

if __name__ == '__main__':
    '''
    exact solution for this problem is:
    W = [-0.5 1]
    b = [0.5]
    '''
    main()