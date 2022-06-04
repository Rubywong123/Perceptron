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

    lemmatizer = WordNetLemmatizer()
    for i in data:
        i['POS'] = nltk.pos_tag(i['words'])
        for idx in i.index:
            if i.iloc[idx, 2][1] in mapping:
                i.iloc[idx, 0] = lemmatizer.lemmatize(i.iloc[idx, 0], pos=mapping[i.iloc[idx, 2][1]])
        i['words'] = [w.lower() for w in i['words']]

def prob_split(data):
    df = pd.DataFrame()
    for i in data:
        df = df.append(i[i['triggers'] == 1])
    df.reset_index(inplace = True)
    df.drop(columns=['index'], inplace=True)
    lm_df = df.groupby('words').count().sort_values(by='triggers', ascending=False)

    triggers = lm_df.index.copy()
    trigger_count = {trigger:0 for trigger in triggers}
    for i in data:
        for idx in i.index:
            if i.iloc[idx, 0] in trigger_count:
                trigger_count[i.iloc[idx, 0]] += 1
    
    trigger_prob = {}
    for word in lm_df.index:
        #print(word, lm_df.loc[word]['triggers'] / trigger_count[word])
        trigger_prob[word] = lm_df.loc[word]['triggers'] / trigger_count[word]
    
    high = []
    very_high = []
    medium = []
    for word in trigger_prob:
        if trigger_prob[word] > 0.5 and trigger_prob[word] <= 0.7:
            medium.append(word)
        if trigger_prob[word] > 0.7 and trigger_prob[word] <= 0.9:
            high.append(word)
        if trigger_prob[word] > 0.9:
            very_high.append(word)
    
    #File Writing
    with open('very_high.txt', 'w') as f:
        for word in very_high:
            f.write(word)
            f.write('\n')
    with open('high.txt', 'w') as f:
        for word in high:
            f.write(word)
            f.write('\n')
    with open('medium.txt', 'w') as f:
        for word in medium:
            f.write(word)
            f.write('\n')





if __name__ == '__main__':
    #remember to connect to VPN.
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    train_data = [pd.DataFrame(i) for i in read_data('train')]
    preprocessing(train_data)
    prob_split(train_data)
    
