# -*- coding: utf-8 -*-
"""

**Task Overview:**
- Implement a basic RNN network to solve time series prediction
- Implement an LSTM network to conduct sentiment analysis

## 1 - Implement a RNN model to predict time series##
### 1.1 Prepare the data (10 Points)

Prepare time series data for deep neural network training.

**Tasks:**
1. Load the given train and test data: "train.txt" and "test.txt". **(2.5 Points)**
2. Generate the **TRAIN** and **TEST** labels. **(2.5 Points)**
3. Normalize the **TRAIN** and **TEST** data with sklearn function "MinMaxScaler". **(2.5 Points)**
4. **PRINT OUT** the **TEST** data and label. **(2.5 Points)**

**Hints:**  
1. The length of original train data is 113 which starts from **"1949-01"** to **"1958-05"**. The length of original test data is 29, which starts from **"1958-07"** to **"1960-11"**.
2. Set the data types of both train and test data to "float32".
3. Use **past 12** datapoints as input data X to predict the **next 1** datapoint as Y, which is the 'next token prediction'. The time window will be 12.
4. The first 3 **TRAIN** data and label should be:

- trainX[0] = [[0.02203858 &nbsp; 0.03856748 &nbsp; 0.077135 &nbsp;  0.06887051 &nbsp; 0.04683197 &nbsp; 0.08539945 &nbsp; 0.12121212 &nbsp; 0.12121212 &nbsp; 0.08815429 &nbsp; 0.04132232 &nbsp; 0.    &nbsp; 0.03856748]]
- trainY[0] = [0.03030303]

- trianX[1] = [[0.03856748 &nbsp; 0.077135 &nbsp;  0.06887051 &nbsp; 0.04683197  &nbsp; 0.08539945  &nbsp; 0.12121212  &nbsp; 0.12121212  &nbsp; 0.08815429  &nbsp; 0.04132232  &nbsp; 0.     &nbsp;  0.03856748   &nbsp; 0.03030303]]
- trainY[1] = [0.06060606]

- trainX[2] =  [[0.077135 &nbsp;  0.06887051 &nbsp; 0.04683197 &nbsp; 0.08539945 &nbsp; 0.12121212 &nbsp; 0.12121212 &nbsp; 0.08815429 &nbsp; 0.04132232 &nbsp; 0.    &nbsp;     0.03856748 &nbsp; 0.03030303 &nbsp; 0.06060606]]
- trainY[2] = [0.10192838]

5. Apply the MinMaxScaler to both the train and test data.\
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Step 1. Use pandas to read training and testing from txt file. 
train_df = pd.read_csv('train.txt')
test_df = pd.read_csv('test.txt')


train_data = train_df['Passengers'].values.astype('float32').reshape(-1, 1)
test_data = test_df['Passengers'].values.astype('float32').reshape(-1, 1)

# Step 2: Normalize the TRAIN and TEST data with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_normalized = scaler.fit_transform(train_data)
test_data_normalized = scaler.transform(test_data)

# Step 3. Create a training and test datasets. 
def dataset(data, time_window):
    X, Y = [], []
    for i in range(len(data) - time_window):
        X.append(data[i:i+time_window])
        Y.append(data[i+time_window])
    return np.array(X), np.array(Y)

# Step 4. Print out the shape of data. 
time_window = 12
trainX, trainY = dataset(train_data_normalized, time_window)
testX, testY = dataset(test_data_normalized, time_window)

# Step 4: Print out the TEST data and labels
for i in range(len(trainX)):
    print(f"trainX[{i}] = {trainX[i].tolist()}")
    print(f"trainY[{i}] = {trainY[i].tolist()}")



for i in range(len(testX)):
    print(f"testX[{i}] = {testX[i].tolist()}")
    print(f"testY[{i}] = {testY[i].tolist()}")

"""###  Build the RNN model  ##


Build a RNN model with RNN cell.


"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Complete the model architecture (10 pts)
class RNN(nn.Module):
     def __init__(self, input_size, hidden_size, output_size):
       super(RNN, self).__init__()
       self.hidden_size = hidden_size
       self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
       self.fc = nn.Linear(hidden_size, output_size)

     def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output


# Create an instance of model, optimizer and criterion. 

input_size = 1
hidden_size = 4
output_size = 1
model = RNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

train_dataset = TensorDataset(torch.Tensor(trainX), torch.Tensor(trainY))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)



# Train the RNN Model for 1000 epoch and print out the trainingloss for every 100 epochs. 
epochs = 1000
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for batch_X, batch_Y in train_loader:
        output = model(batch_X)
        loss = criterion(output, batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 100 == 0:
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}], Average Training Loss: {average_loss:.4f}")




# Step 1. Inference above model on training and testing data. 
with torch.no_grad():
    train_predictions = model(torch.Tensor(trainX)).squeeze().numpy()
    test_predictions = model(torch.Tensor(testX)).squeeze().numpy()

# Step 2. Denomalization. (3 pts)
train_predictions_denormalized = scaler.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
test_predictions_denormalized = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()

# Step 3. Calculate root mean squared error for training and testing and print.
def calculate_rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

train_rmse = calculate_rmse(train_predictions_denormalized, train_df['Passengers'][time_window:].values)
test_rmse = calculate_rmse(test_predictions_denormalized, test_df['Passengers'][time_window:].values)

print(f"TRAIN RMSE: {train_rmse:.2f}")
print(f"TEST RMSE: {test_rmse:.2f}")
# Step 4. Plot the predictions. (2 pts)

plt.plot(test_df['Passengers'][:len(test_predictions_denormalized)], label='Actual')
plt.plot(test_predictions_denormalized, label='Predicted')
plt.show()

"""## 2 - Use LSTM model to conduct sentiment analysis ##


"""

import torch
import random
import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

# Step 1. Load IMDB dataset from keras.
max_words = 1000
max_length = 100
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Print out the lengths of sequences
print("Length of sequences in training data:")
for sequence in x_train:
    print(len(sequence))


# Step 2. Preprocess the sequences with padding 
maxlen = 100
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

print("Shape of x_train after padding:", x_train.shape)
print("Shape of x_test after padding:", x_test.shape)

"""###  Design and train LSTM model  ###

Build a LSTM model.


"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences


max_words = 1000
max_length = 100
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

x_train_tensor = torch.tensor(x_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float)
x_test_tensor = torch.tensor(x_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.float)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Complete LSTM model architecture. 
class LSTMClassifier(nn.Module):
     def __init__(self, max_features, embedding_dim, hidden_dim, num_layers):
        super(LSTMClassifier, self).__init__()
        self.embeddings = nn.Embedding(max_features, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

     def forward(self, x):
        embeds = self.embeddings(x)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :]
        output = self.classifier(lstm_out)
        output = self.sigmoid(output)
        return output


max_features = 1000
embedding_dim = 8
hidden_dim = 8
num_layers = 1
model = LSTMClassifier(max_features, embedding_dim, hidden_dim, num_layers)
# Create an instance of LSTM model, an adam optimizer and BCE loss. 

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())


# Train the model, print out the loss. 
best_accuracy = 0.0
num_epochs = 10

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

    val_accuracy = 100 * correct / total
    if (val_accuracy>best_accuracy):
      best_accuracy = val_accuracy

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Train Accuracy: {train_accuracy:.4f}%, "
          f"Val Accuracy: {val_accuracy:.4f}%")



# Find best validation accuracy

print(f"Best Validation Accuracy: {best_accuracy:.4f}%")

"""# Reinforcement Learning



Episode 1, Step1: o----T   \
... \
Episode 1, Step6: ---o-T   \
... \
Episode 1, Step10: -o---T \
... \
Episode 1, Step15: ----oT (finished) \




- Train the explorer getting the treasure quickly through Q-learning method

 Achieve Q-learning method 



"""

import numpy as np
import pandas as pd
import time

N_STATES = 6   # the width of 1-dim world
ACTIONS = ['left', 'right']     # the available actions to use
EPSILON = 0.9   # the degree of greedy (0＜ε＜1)
ALPHA = 0.1     # learning rate (0＜α≤1)
GAMMA = 0.9    # discount factor (0＜γ＜1)
MAX_EPOCHES = 13   # the max epoches
FRESH_TIME = 0.3    # the interval time

"""

Q table is a [states * actions] matrix, which stores Q-value of taking one action in that specific state. For example, the following Q table means in state s3, it is more likely to choose a1 because it's Q-value is 5.31 which is higher than Q-value 2.33 for a0 in s3
![](https://drive.google.com/uc?export=view&id=1WGh7NYyYw6ccrxbDVdfbJmb_IhBfUyFf)

**Tasks:**
1. define the build_q_table function
2. **Print Out** defined Q-table. The correct print information should be:

|     | left | right |
|-----|------|-------|
| 0   | 0.0  | 0.0   |
| 1   | 0.0  | 0.0   |
| 2   | 0.0  | 0.0   |
| 3   | 0.0  | 0.0   |
| 4   | 0.0  | 0.0   |
| 5   | 0.0  | 0.0   |


    


"""

#define the function here
def build_q_table(n_states, actions):
    q_table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return q_table

q_table = build_q_table(N_STATES, ACTIONS)
print(q_table)

"""
"""

#define the function here
# Given state and Q-table, choose action
def choose_action(state, q_table):
#       # pick all actions from this state
  state_actions = q_table.iloc[state, :]
  if (np.random.uniform() < (1 - EPSILON)) or (state_actions.all() == 0):  # Non-greedy or non-explored
      action_name = np.random.choice(ACTIONS)
  else:  # Greedy
      action_name = state_actions.idxmax()

  return action_name

sample_action = choose_action(0, q_table)
print(sample_action)

"""

In this section, we need to give a feedback for our previous action, which means getting reward (R) for next state (S_next) based on current state (S_current) and action (A). In this problem, we get reward R=1 if we move to the treasure T spot, otherwise, we get R=0.



- S_current=0, sample_action = 'right', sample_feedback=(1,0)
- S_current=3, sample_action = 'right', sample_feedback=(4,0)
- S_current=4, sample_action = 'right', sample_feedback=('terminal', 1)
- S_current=0, sample_action = 'left', sample_feedback=(0,0)
- S_current=3, sample_action = 'left', sample_feedback=(2,0)
- S_current=4, sample_action = 'left', sample_feedback=(3, 0)
"""

#define the function here
def get_env_feedback(S_current, A):
#     # This is how agent will interact with the environment
  if A == 'right':    # move right
    if S_current == N_STATES - 2:  # terminate if the next state is the terminal state
            S_next = 'terminal'
            R = 1
    else:
            S_next = S_current + 1
            R = 0
  else:   # move left
    if S_current == 0:  # boundary
            S_next = S_current  # reach the wall
    else:
            S_next = S_current - 1
    R = 0
  return S_next, R

sample_action = 'left'
S_current = 4
sample_feedback = get_env_feedback(S_current, sample_action)
print(sample_feedback)

def update_env(S, episode, step_counter):
#     # This is how environment be updated
     env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
     if S == 'terminal':
         interaction = '  Episode %s: total_steps = %s' % (episode+1, step_counter)
         print('{}\n'.format(interaction), end='')
         time.sleep(2)
     else:
         env_list[S] = 'o'
         interaction = ''.join(env_list)
         print('\r{}'.format(interaction), end='')
         time.sleep(FRESH_TIME)

"""### Start Q-learning with defined functions 
"""



#define the function here
def reinforce_learning():
  q_table = build_q_table(N_STATES, ACTIONS)
#     # main part of RL loop
#     # build Q-table here
#     ...
#     #start training loop
  for episode in range(MAX_EPOCHES):
         step_counter = 0  #counter for counting steps to reach the treasure
         S_current = 0     #start from S_current
         is_terminated = False   #flag to conrinue or stop the loop
         update_env(S_current, episode, step_counter)   #update environment
         while not is_terminated:
#             ...#choose one action
            A = choose_action(S_current, q_table)
#             ...# take action & get next state and reward
            S_next, R = get_env_feedback(S_current, A)
#             ...#update Q-table
            if S_next != 'terminal':                   #if the explorer doesn't get to the treasure
                 q_target = R + GAMMA * q_table.iloc[S_next, :].max()  #Bellman eq   # if next state is not terminal, how can we estimate the q value (hit: bellman equation)?
            else:
                 q_target = R     # if next state is terminal, how can we esimate the q value?
                 is_terminated = True    # terminate this episode

            q_table.loc[S_current, A] += ALPHA * (q_target - q_table.loc[S_current, A])  # update Q-table
#             ...  # move to next state
            S_current = S_next

            update_env(S_current, episode, step_counter+1)
            step_counter += 1
  return q_table

#main function to run
if __name__ == "__main__":
     q_table = reinforce_learning()
     print('\r\nQ-table:\n')
     print(q_table)

