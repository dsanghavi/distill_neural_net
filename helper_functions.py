import numpy as np

def accuracy_in_batches(data, accuracy_func, x, y_, keep_prob=None, batch_size=50):
    y_accs = []
    for j in range(data.num_examples/batch_size):
        check_batch = data.next_batch(batch_size)
        if keep_prob is None:
            y_accs.append(accuracy_func.eval(feed_dict={
                    x: check_batch[0], y_: check_batch[1]}))
        else:
            y_accs.append(accuracy_func.eval(feed_dict={
                    x: check_batch[0], y_: check_batch[1], keep_prob: 1.0}))
    return np.mean(y_accs)

# def cross_entropy_loss(x, y):
#     N = x.shape[0]
#     loss = (-1/N) *
#     return loss