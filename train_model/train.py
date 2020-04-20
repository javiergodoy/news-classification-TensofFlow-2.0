import argparse
import numpy as np
import os
import tensorflow as tf
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

from model_def import get_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')


def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=5000)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    
    return parser.parse_known_args()


def get_train_data(train):
    
    x_train = np.load(os.path.join(train, 'x_train.npy'))
    y_train = np.load(os.path.join(train, 'y_train.npy'))
    print('x train', x_train.shape,'y train', y_train.shape)

    return x_train, y_train


def get_test_data(test):
    
    x_test = np.load(os.path.join(test, 'x_test.npy'))
    y_test = np.load(os.path.join(test, 'y_test.npy'))
    print('x test', x_test.shape,'y test', y_test.shape)

    return x_test, y_test
   

if __name__ == "__main__":
        
    args, _ = parse_args()
    
    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.test)
    vocab_size = args.vocab_size
    embedding_dim = args.embedding_dim 
    
    device = '/cpu:0' 
    print(device)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))

    with tf.device(device):
        
        #model = get_model(vocab_size, embedding_dim)

        # evaluate on test set
        #scores = model.evaluate(x_test, y_test, batch_size, verbose=2)
        #print("\nTest MSE :", scores)
        
        # save model
        #model.save(args.model_dir + '/1')
        #tf.contrib.saved_model.save_keras_model(model, /opt/ml/model)

        #tf.contrib.saved_model.save_keras_model(model, args.model_dir)
        
        model = get_model(vocab_size, embedding_dim)

        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

        model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(x_test, y_test))

        # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
        model.save('args.model_dir')
        with open(os.path.join(model_path, 'xgboost-model.pkl'), 'wb') as out:
            pickle.dump(model, out, protocol=0)
        print('Training complete.')