from model import StructuredPerceptron, Perceptron
import numpy as np
from util import read_data
from features import generate_feature_matrix

def main():
    train_data = read_data('train')
    valid_data = read_data('valid')
    test_data = read_data('test')

    example_features = generate_feature_matrix(train_data[0]['words'])
    n_classes = 2
    n_features = 9
    model = Perceptron(n_classes, n_features)

    model.train(train_data)
    print(model.weights)

    # test
    model.eval(test_data)



if __name__ == '__main__':
    main()