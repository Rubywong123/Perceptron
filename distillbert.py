from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification
from torch.optim.lr_scheduler import StepLR, LinearLR
from util import read_data
from datasets import load_dataset, Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
def load_hf_dataset():
    train_data = read_data('train')
    valid_data = read_data('valid')
    test_data = read_data('test')

    train_set = {'tokens': [data['words'] for data in train_data],'triggers': [data['triggers'] for data in train_data]}
    valid_set = {'tokens': [data['words'] for data in valid_data],'triggers': [data['triggers'] for data in valid_data]}
    test_set = {'tokens': [data['words'] for data in test_data],'triggers': [data['triggers'] for data in test_data]}

    train_dataset = Dataset.from_dict(train_set)
    valid_dataset = Dataset.from_dict(valid_set)
    test_dataset = Dataset.from_dict(test_set)

    return train_dataset, valid_dataset, test_dataset



'''using DistillBERT to do fine-tuning and inference'''
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words = True, truncation = True)

    labels = []
    for i, label in enumerate(examples['triggers']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    
    return tokenized_inputs

def to_device(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)


def main():
    # can be modified
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    #loading datasets
    train_dataset, valid_dataset, test_dataset = load_hf_dataset()
    label_list = [0, 1]

    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched = True, remove_columns=['tokens', 'triggers'])
    tokenized_valid = valid_dataset.map(tokenize_and_align_labels, batched = True, remove_columns=['tokens', 'triggers'])
    tokenized_test = test_dataset.map(tokenize_and_align_labels, batched = True, remove_columns=['tokens', 'triggers'])


    data_collator = DataCollatorForTokenClassification(tokenizer = tokenizer)

    model = AutoModelForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels = 2)
    
    model.to(device)
    
    train_dl = DataLoader(tokenized_train, batch_size = 16, collate_fn = data_collator, shuffle = True)
    valid_dl = DataLoader(tokenized_valid, batch_size = 16, collate_fn = data_collator, shuffle = True)
    test_dl = DataLoader(tokenized_test, batch_size = 16, collate_fn = data_collator, shuffle = True)

    #Optimizer

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)

    lr_scheduler = LinearLR(optimizer)


    #training
    print('Training...')
    train_epochs = 3
    for epoch in range(train_epochs):
        print('====== Epoch {} ======'.format(epoch))
        model.train()
        total_loss = 0
        step = 0
        for batch in train_dl:
            optimizer.zero_grad()
            to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0:
                print(" | batch loss: %.6f step = %d" % (loss.item(), step))
            total_loss += loss.item()
        print('| epoch %d avg loss = %.6f' % (epoch, total_loss / step))
        lr_scheduler.step()

        #evaluation
        print('=====================evaluation=====================')
        model.eval()
        total_loss = 0
        step = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        for batch in valid_dl:
            with torch.no_grad():
                to_device(batch, device)

                outputs = model(**batch)
                total_loss += outputs.loss.item()
                step += 1
                predictions = outputs.logits.argmax(dim=-1)

                #compute metrics
                for i in range(len(batch)):
                    for j in range(len(batch['labels'][i])):
                        if batch['labels'][i][j] == 0:
                            if predictions[i][j] == 0:
                                TP += 1
                            else:
                                FN += 1
                        else:
                            if predictions[i][j] == 0:
                                TN += 1
                            else:
                                FP += 1
        print('| epoch %d avg val_loss = %.6f' % (epoch, total_loss / step))
        print(TP, TN, FP, FN)
        print('precision=%.6f, recall=%.6f, f1=%.6f' % (TP/(TP+FP), TP/(TP+FN), 2*TP/(2*TP+FP+FN)))

        #inference
    print('======Test======')
    model.eval()
    TP, TN, FP, FN = 0, 0, 0, 0
    for batch in test_dl:
        with torch.no_grad():
            to_device(batch, device)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            #compute metrics
            for i in range(len(batch)):
                for j in range(len(batch['labels'][i])):
                    if batch['labels'][i][j] == 0:
                        if predictions[i][j] == 0:
                            TP += 1
                        else:
                            FN += 1
                    else:
                        if predictions[i][j] == 0:
                            TN += 1
                        else:
                            FP += 1
    print(TP, TN, FP, FN)
    print('precision=%.6f, recall=%.6f, f1=%.6f' % (TP/(TP+FP), TP/(TP+FN), 2*TP/(2*TP+FP+FN)))



if __name__ == '__main__':
    main()

