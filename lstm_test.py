import torch
import torch.nn as nn
from torch.autograd import Variable

time_steps = 10
batch_size = 3
in_size = 5
classes_no = 7

model = nn.LSTM(in_size, classes_no, 2)
input_seq = Variable(torch.randn(time_steps, batch_size, in_size))
print(input_seq.shape)
output_seq, _ = model(input_seq)
print(output_seq.shape)
last_output = output_seq[-1]

loss = nn.CrossEntropyLoss()
target = Variable(torch.LongTensor(batch_size).random_(0, classes_no-1))
err = loss(last_output, target)
err.backward()