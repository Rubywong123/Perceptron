import json
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd


'''
return:
data[index] --> {'words': list of words, 'triggers': list of triggers(0-1 labelled)]}
'''
def read_data(split_name):
    path = 'data/'+split_name+'.json'
    with open(path, 'r') as f:
        return json.load(f)

'''Is it necessary to convert it into BIO format?'''

mapping = {'JJR': 'a', 'JJS': 'a', 'NNS': 'n', 'NNPS': 'n', 
            'RBR': 'r', 'RBS': 'r', 'VBD': 'v', 'VBG': 'v',
            'VBN': 'v', 'VBP': 'v', 'VBZ': 'v'
}
def preprocessing(data):
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    for i in data:
        i['POS'] = nltk.pos_tag(i['words'])
        for idx in i.index:
            if i.iloc[idx, 2][1] in mapping:
                i.iloc[idx, 0] = lemmatizer.lemmatize(i.iloc[idx, 0], pos=mapping[i.iloc[idx, 2][1]])
        i['words'] = [w.lower() for w in i['words']]




if __name__ == '__main__':
    train_data = [pd.DataFrame(i) for i in read_data('train')]
    df = pd.DataFrame()
    for i in train_data:
        #print(i[i['triggers'] == 1])
        df = df.append(i[i['triggers'] == 1])
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    df.groupby('words').count().sort_values(by='triggers', ascending=False)
    df2 = df.copy()
    lemmatizer = WordNetLemmatizer()
    l = df2['words'].to_list()
    l = [lemmatizer.lemmatize(word) for word in l]
    print(l)
