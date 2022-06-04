import pandas as pd
from util import read_data, preprocessing


train_data = [pd.DataFrame(i) for i in read_data('train')]
preprocessing(train_data)

df = pd.DataFrame()
for i in train_data:
    df = df.append(i[i['triggers'] == 1])
df.reset_index(inplace=True)
df.drop(columns=['index'], inplace=True)
lm_df = df.groupby('words').count().sort_values(by='triggers', ascending=False)
triggers = lm_df.index.copy()
trigger_count = {trigger: 0 for trigger in triggers}
for i in train_data:
    for idx in i.index:
        if i.iloc[idx, 0] in trigger_count:
            trigger_count[i.iloc[idx, 0]] += 1
trigger_prob = {}
sum = 0
for word in lm_df.index:
    trigger_prob[word] = lm_df.loc[word]['triggers']/trigger_count[word]
    sum += lm_df.loc[word]['triggers']/trigger_count[word]

test_data = [pd.DataFrame(i) for i in read_data('test')]
preprocessing(test_data)
TP = 0
FP = 0
FN = 0
TN = 0
for i in test_data:
    for idx in i.index:
        if i.iloc[idx, 0] in triggers and trigger_prob[i.iloc[idx, 0]] >= 0.5: 
            if i.iloc[idx, 1] == 1:
                TP += 1
            else:
                FP += 1
        elif i.iloc[idx, 1] == 1:
            FN += 1
        elif i.iloc[idx, 1] == 0:
            TN += 1

print(TP, FP, FN, TN)
print('Precision: ', TP / (TP + FP))
print('Recall: ', TP / (TP + FN))
print('F1: ', 2 * TP / (2 * TP + FP + FN))
