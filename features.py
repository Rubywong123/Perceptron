import nltk
from nltk.parse.stanford import StanfordDependencyParser
from nltk.stem import WordNetLemmatizer
import numpy as np

mapping = {'JJ':'a', 'JJR': 'a', 'JJS': 'a', 'NNS': 'n', 'NNPS': 'n', 
            'RBR': 'r', 'RBS': 'r', 'VBD': 'v', 'VBG': 'v',
            'VBN': 'v', 'VBP': 'v', 'VBZ': 'v', 'NN' : 'n',
            'VB': 'v'

}

'''return a string of pos tags of their corresponding words. Containing punctuations.'''
def get_pos(tokens):
    pos_tag = nltk.pos_tag(tokens)
    result = []
    for word, pos in pos_tag:
        result.append(pos)
    return result


'''
NER
return: a tree stands for NER labels.
for word in ners:
    if type(word) == nltk.tree.Tree:
        [further operations]
'''
def get_ner(tokens):
    nltk.download('maxent_ne_chunker')
    tags = nltk.pos_tag(tokens)
    ners = nltk.ne_chunk(tags)

    return ners

'''
dependency parsing
return: list of dependencies(in the form of tuples)
[((word1, pos1), relation, (word2, pos2)), ...]
[head, relation, dependent]
'''


def get_dep(sentence):
    '''Warning! 2 parser models are needed to be downloaded and put in the path.'''
    path_to_jar = 'parser/stanford-parser.jar'
    path_to_models_jar = 'parser/stanford-parser-4.2.0-models.jar'
    parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

    result = parser.raw_parse(sentence)
    dep = next(result)
    return list(dep.triples())


'''generate feature matrix'''


'''4 ugram features
1) word unigram, aims at using tokens that are highly related to being triggers.
2) dependency parser, the head of a dependency tree tends to be a trigger.
return: (L, 4) matrix
'''
def get_ugram_features(tokens):
    num_features = 3
    num_classes = 2
    ugram_features = np.zeros((len(tokens), num_classes, num_features))
    ### load lists of triggers with very high/ high/ medium frequency.
    with open('very_high.txt', 'r') as f:
        very_high = f.read().splitlines()
    with open('high.txt', 'r') as f:
        high = f.read().splitlines()
    with open('medium.txt', 'r') as f:
        medium = f.read().splitlines()
    with open('good_pair.txt', 'r') as f:
        pair = f.read().splitlines()
    lemmatizer = WordNetLemmatizer()
    #sentence = ' '.join(tokens)
    #dep = get_dep(sentence)

    #generate features
    pos = ['n', 'v', 'r', 'a']
    token2index = {}
    for i, token in enumerate(tokens):
        token2index[token] = i
        for p in pos:
            lemma = lemmatizer.lemmatize(token, pos = p)
            if lemma in very_high:
                ugram_features[i, 0, 0] = 1
                ugram_features[i, 1, 0] = 1
                break
            elif lemma in high:
                ugram_features[i, 0, 1] = 1
                ugram_features[i, 1, 1] = 1
                break
            elif lemma in medium:
                ugram_features[i, 0, 2] = 1
                ugram_features[i, 1, 2] = 1
                break

        '''if i > 0:
            before_token = tokens[i-1]
            for p1 in pos:
                for p2 in pos:
                    lemma = lemmatizer.lemmatize(token, pos = p1)
                    lemma_before = lemmatizer.lemmatize(before_token, pos = p2)
                    phrase = lemma_before + ' ' + lemma
                    if phrase in pair:
                        ugram_features[i, 0, 3] = 1
                        ugram_features[i, 1, 3] = 1
        if i < len(tokens)-1:
            after_token = tokens[i+1]
            for p1 in pos:
                for p2 in pos:
                    lemma = lemmatizer.lemmatize(token, pos = p1)
                    lemma_after = lemmatizer.lemmatize(after_token, pos = p2)
                    phrase = lemma + ' ' + lemma_after
                    if phrase in pair:
                        ugram_features[i, 0, 3] = 1
                        ugram_features[i, 1, 3] = 1
        '''    


    #dependency head
    '''
    for edge in dep:
        word = edge[0][0]
        dependent = edge[2][0]
        word_pos = edge[0][1]
        dep_pos = edge[2][1]
        if word in token2index:
            if word_pos in mapping:
                if mapping[word_pos] == 'v':
                    ugram_features[token2index[word]][0][3] = 1
                    ugram_features[token2index[word]][1][3] = 1
                elif mapping[word_pos] == 'n':
                    ugram_features[token2index[word]][0][4] = 1
                    ugram_features[token2index[word]][1][4] = 1
                else:
                    ugram_features[token2index[word]][0][5] = 1
                    ugram_features[token2index[word]][1][5] = 1
            else:
                ugram_features[token2index[word]][0][5] = 1
                ugram_features[token2index[word]][1][5] = 1
        if dependent in token2index:
            if dep_pos in mapping:
                if mapping[dep_pos] == 'v':
                    ugram_features[token2index[dependent]][0][6] = 1
                    ugram_features[token2index[dependent]][1][6] = 1
                elif mapping[dep_pos] == 'n':
                    ugram_features[token2index[dependent]][0][7] = 1
                    ugram_features[token2index[dependent]][1][7] = 1
                else:
                    ugram_features[token2index[dependent]][0][8] = 1
                    ugram_features[token2index[dependent]][1][8] = 1
            else:
                ugram_features[token2index[dependent]][0][8] = 1
                ugram_features[token2index[dependent]][1][8] = 1
    '''
    
    #return np.concatenate([ugram_features[:,:,0], ugram_features[:,:,1]], axis=1)
    return ugram_features

'''bigram features
1) 
'''
def get_bigram_features(tokens, triggers):
    pass


'''
return (L, n_classes * features_per_label) matrix
'''
def generate_feature_matrix(tokens):
    ugram_features = get_ugram_features(tokens)
    '''bigram features
    
    '''

    '''np.concatenate((ugram_features, bigram_features), axis = 1)'''
    features = ugram_features

    return features



if __name__ == '__main__':
    tokens = ['I', 'have', 'attended', 'a', 'war']
    triggers = [0, 1, 0, 0, 1]
    features = generate_feature_matrix(tokens)

    print(features)