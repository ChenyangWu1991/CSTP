# %% [markdown]
# # Predict the sentiment of all sentences by BERT

# %% [markdown]
# ## Read data (sentences of the full text of the FYP-Ts of 23 cities)

# %%
import pandas as pd

all_data = pd.read_csv('all_23_cities.csv')
print(all_data.shape)
all_data.head(1)

# %% [markdown]
# ## Data preprocessing (filter sentences by tokens)

# %%
all_data_token = all_data[all_data['sum'] > 0]
all_data_same_senti_label = all_data[all_data['Gap'] == 0]
all_data_token_same_senti_label = all_data_token[all_data_token['Gap'] == 0]

# %%
print(all_data_token.shape)
print(all_data_same_senti_label.shape)
print(all_data_token_same_senti_label.shape)

# %%
all_data_token_same_senti_label = all_data_token_same_senti_label.astype({"SA_Li": 'int64', "SA_Guo": 'int64', 'Gap': 'int64'})
# all_data_token_same_senti_label.dtypes

# %% [markdown]
# ## Train test data split

# %%
train = all_data_token_same_senti_label.sample(n=600, random_state=33)
# exclude the training set
test = pd.concat([all_data_token, train, train]).drop_duplicates(keep=False)  # sentence level
# test = pd.concat([all_data, train, train]).drop_duplicates(keep=False).sample(n=1000, random_state=33)  # article level
print(train.shape)
print(test.shape)
print(all_data_token.shape)

# %%
print(train['Token'].values[0])
print('- - - - - - - - - - - - -')
print(test['Token'].values[0])
train_x = train['Token'].tolist()
train_y = list(map(lambda x:x+1, train['SA_Li'].tolist()))
test_x = test['Token'].tolist()
test_y = test['SA_Li'].tolist()

# %% [markdown]
# ## Use transformer to define the fine-tune model based on BERT

# %%
import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn

model_name = 'hfl/chinese-bert-wwm'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# A bert fine-tuning strategy is used to adjust the parameters of the BERT and the linear layer together during back propagation to make BERT more suitable for the classification task.
class BertClassfication(nn.Module):
    def __init__(self):
        super(BertClassfication,self).__init__()
        self.model_name = 'hfl/chinese-bert-wwm'
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768,3)     # depend on the structure of BERT, 2-layer, 768-hidden, 12-heads, 110M parameters
        # nn.Linear(in_features, out_features)

    def forward(self,x):               # The input is a list here.
        batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
                                max_length=148, pad_to_max_length=True)     
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        hiden_outputs = self.model(input_ids,attention_mask=attention_mask)
        outputs = hiden_outputs[0][:,0,:]     
        output = self.fc(outputs)
        return output
model = BertClassfication()

# %% [markdown]
# ## Start training with batches

# %%
batch_size = 64
batch_count = int(len(train) / batch_size)
batch_train_inputs, batch_train_targets = [], []

for i in range(batch_count):
    batch_train_inputs.append(train_x[i*batch_size : (i+1)*batch_size])
    batch_train_targets.append(train_y[i*batch_size : (i+1)*batch_size])

# 初始化训练参数
bertclassfication = BertClassfication()
lossfuction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(bertclassfication.parameters(),lr=2e-5)
epoch = 5
batch_count = batch_count
print_every_batch = 5

# %%
for _ in range(epoch):
    print_avg_loss = 0
    for i in range(batch_count):
        inputs = batch_train_inputs[i]
        targets = torch.tensor(batch_train_targets[i])
        optimizer.zero_grad()
        outputs = bertclassfication(inputs)
        loss = lossfuction(outputs, targets)
        loss.backward()
        optimizer.step()

        print_avg_loss += loss.item()
        if i % print_every_batch == (print_every_batch-1):
            print("Batch: %d, Loss: %.4f" % ((i+1), print_avg_loss/print_every_batch))
            print_avg_loss = 0

# %% [markdown]
# ## Predict with the fine-tune model

# %% [markdown]
# ### Try a demo

# %%
# sentiment_dict = {0:'Negative', 1:'Neutral', 2:'Positive'}

result = bertclassfication([train_x[0]])
_, predict = torch.max(result,1)
print(train_y[0])

# %% [markdown]
# ### The results of the training set by BERT

# %%
output_train = []

for i in train_x:
    result = bertclassfication([i])
    _, predict = torch.max(result,1)
    output_train.append(int(predict))

# %%
train['SA_BERT'] = output_train
train['SA_BERT'] -= 1
# train.to_csv('training_set.csv', encoding='utf_8_sig', index=None)

# %% [markdown]
# ### The results of the test set by BERT

# %%
output_test = []

for i in test_x:
    result = bertclassfication([i])
    _, predict = torch.max(result,1)
    output_test.append(int(predict))

# %%
test['SA_BERT'] = output_test
test['SA_BERT'] -= 1
# test.to_csv('sentence_level_test.csv', encoding='utf_8_sig', index=None)
# test.to_csv('article_level_test.csv', encoding='utf_8_sig', index=None)


