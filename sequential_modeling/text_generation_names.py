import json
import pandas as pd
import urllib.request

from __future__ import unicode_literals, print_function, division
import unicodedata
import string

import torch
import torch.nn as nn

import random
import time
import math


def convertASCII(word):
    asciiWord = ""
    word = str(word).rstrip()
    for c in word:
        if c in characterMap.keys():
            asciiWord = asciiWord + characterMap[c]
        else:
            asciiWord = asciiWord + c
    return asciiWord;


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device='cuda')

def train(category_tensor, input_line_tensor, target_line_tensor, optimizer):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    optimizer.zero_grad()


    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=2.0, norm_type=2)
    optimizer.step()

    loss_val = loss.detach().cpu().numpy() / input_line_tensor.size(0)

    return output, loss_val

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()
        hidden = hidden.to('cpu')

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))



all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker

list_names = ['Arabic', 'Chinese', 'Czech', 'Dutch',
              'English', 'French', 'German', 'Greek',
              'Irish', 'Italian', 'Japanese', 'Korean',
              'Polish', 'Portuguese', 'Russian', 'Scottish',
              'Spanish', 'Vietnamese']


data = []
for ethnicity in list_names:
    url = "https://github.com/cmlakhan/ml_code/raw/master/sequential_modeling/data/names/{}.txt".format(ethnicity)
    file = urllib.request.urlopen(url)

    for line in file:
        data.append({'name': line.decode('utf-8'), 'ethnicity':ethnicity})


names_df = pd.DataFrame(data)
all_categories = names_df.ethnicity.unique().tolist()
n_categories = len(all_categories)


names_df[['name_ascii']] = names_df.name.apply(lambda x: unicodeToAscii(x))

names_df.head()




rnn = RNN(n_letters, 512, n_letters)
criterion = nn.NLLLoss()
learning_rate = 0.0005
frac = 1.00
num_epochs = 2

print_every = 1000
plot_every = 50
all_losses = []
total_loss = 0 # Reset every plot_every iters


start = time.time()
device = 'cuda'

rnn.to(device)

optimizer = torch.optim.AdamW(rnn.parameters(), lr=learning_rate, weight_decay=0.00001, amsgrad=False)


iter = 0
for i in range(1,num_epochs+1):
    print('Epoch #{}'.format(i))
    subset_df = names_df.sample(frac=frac)
    len_df = subset_df.shape[0]
    for index,row in subset_df.iterrows():
        iter += 1
        category_tensor = categoryTensor(row.ethnicity)
        input_line_tensor = inputTensor(row.name_ascii)
        target_line_tensor = targetTensor(row.name_ascii)
        category_tensor = category_tensor.to(device)
        input_line_tensor = input_line_tensor.to(device)
        target_line_tensor = target_line_tensor.to(device)
        output, loss = train(category_tensor, input_line_tensor, target_line_tensor, optimizer)
        total_loss += loss
        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / len_df * 100, loss))
        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0




import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)



max_length = 20
rnn.to('cpu')
samples('Arabic', 'RUS')

