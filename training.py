import pickle

import tensorflow as tf
import tflearn

import data

train_x, train_y, words, classes = data.get_training_data()

# Reset underlying graph data
tf.reset_default_graph()


def build_model():
    # Init model
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

    return model


if __name__ == '__main__':
    model = build_model()
    model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')
    # save all data structures
    pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y},
                open("training_data", "wb"))
