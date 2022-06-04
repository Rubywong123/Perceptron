import numpy as np
from util import read_data
from features import generate_feature_matrix

class Perceptron:
    '''Perceptron'''
    def __init__(self, nb_classes, nb_features):
        '''Constructor'''
        self.n_classes = nb_classes
        self.n_features = nb_features
        # one lambda for each feature
        self.weights = np.zeros((nb_features,nb_classes))

    '''
    predict(with decoding)
    Return: (L,) array

    '''
    def predict(self, x):
        ans = np.zeros((len(x),))
        for i in range(len(x)):
            zero_score = x[i, 0, :].dot(self.weights[:, 0])
            one_score = x[i, 1, :].dot(self.weights[:, 1])
            if one_score > zero_score:
                ans[i] = 1
        return ans


    def update(self, features, triggers, prediction):
        for i in range(len(features)):
            if triggers[i] != prediction[i]:
                gold = triggers[i]
                #print('before update', self.weights)
                self.weights[:, gold] = self.weights[:, gold] + features[i, gold, :]
                self.weights[:, (1-gold)] = self.weights[:, (1-gold)] - features[i, (1-gold), :]
                #print('after update', self.weights)
        
        

    def train(self,train_data):
        for step, instance in enumerate(train_data):
            
            tokens = instance['words']
            triggers = instance['triggers']
            features = generate_feature_matrix(tokens)
            prediction = self.predict(features)
            self.update(features, triggers, prediction)
            if step % 1000 == 0:
                print('=========step:', step)
                print(self.weights)


    def eval(self, test_data):
        TP, FP, FN = 0, 0, 0
        for step, instance in enumerate(test_data):
            tokens = instance['words']
            triggers = instance['triggers']
            features = generate_feature_matrix(tokens)
            prediction = self.predict(features)

            
            for i in range(len(triggers)):
                if triggers[i] == 1:
                    if prediction[i] == 1:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if prediction[i] == 1:
                        FP += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print('precision: {}, recall: {}, f1: {}'.format(precision, recall, f1))
        
        

'''Structured Perceptron'''

class StructuredPerceptron:
    '''Structured Perceptron'''
    def __init__(self, nb_classes, nb_features):
        '''Constructor'''
        self.n_classes = nb_classes
        self.n_features = nb_features
        self.weights = [0] * nb_features

    '''
    predict(with decoding)
    Return: (L, 2) matrix

    '''
    def predict(self, x):
        pass

    def update(self, x, y, prediction):
        pass

    def train(self, X, Y):
        pass



if __name__ == '__main__':
    model = Perceptron(2,5)
    train_data = read_data('train')
    model.train(train_data)