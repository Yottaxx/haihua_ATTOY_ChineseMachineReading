import json

import pandas as pd
from transformers import BertTokenizer

with open('validation.json', 'r', encoding='utf-8')as f:
    test_data = json.load(f)

test_df = []

for i in range(len(test_data)):
    data = test_data[i]
    content = data['Content']
    questions = data['Questions']
    cls = data['Type']
    diff = data['Diff']

    for question in questions:
        question['Content'] = content
        question['Type'] = cls
        question['Diff'] = diff
        question['Q_id'] = str(question['Q_id'])
        test_df.append(question)
        print(question['Q_id'])

test_df = pd.DataFrame(test_df)
test_df.to_csv('test.csv', index=False)
test_df = pd.read_csv('test.csv')
print(test_df['Q_id'])

# tokenizer = BertTokenizer.from_pretrained('clue/roberta_chinese_3L768_clue_tiny')


# def getQuestionInputs(lines):
#     print(lines['Content'])
#     print(len(lines['Content']))
#     print(len(lines['Content'][0]))
#
#     lines['Content'] = tokenizer(lines['Content'], padding='max_length', truncation=True, max_length=512)
#     lines['input_ids'] = lines['Content']['input_ids']
#     lines['attention_mask'] = lines['Content']['attention_mask']
#     return lines[['input_ids', 'attention_mask', 'Answer']]
#
# def contactContentQuestion(lines):
#     lines['opA'] =lines['question'] + lines['opA'] + tokenizer.sep_token + lines['passage']
#     lines['opB'] = lines['question'] + lines['opB'] + tokenizer.sep_token + lines['passage']
#     lines['opC'] = lines['question'] + lines['opC'] + tokenizer.sep_token + lines['passage']
#     lines['opD'] = lines['question'] + lines['opD'] + tokenizer.sep_token + lines['passage']
#     lines['Content'] = [lines['opA'], lines['opB'], lines['opC'], lines['opD']]
#     lines['Answer']  = ord(lines['answer']) - ord('A')
#
#
#     return lines[['Content', 'Answer']]
#
# data = data.apply(lambda x:contactContentQuestion(x),axis=1)
# data = data.apply(lambda x:getQuestionInputs(x),axis =1 )
