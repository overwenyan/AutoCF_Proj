import os
import sys
import random

from cProfile import label
from cmath import cosh, sin
from tkinter.tix import Tree
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# from bore.models import MaximizableSequential
# from tensorflow.keras.layers import Dense


class FuncBlackBox():
    '''A simple sin function for fitting'''
    def __init__(self, feature_dim) -> None:
        self.feature_dim = feature_dim
        
    def evaluate(self, x):
        if self.feature_dim == 1:
            y = sin(3*x) + x**2 - 0.7*x
        else:
            y = sin(3*x) + x**2 - 0.7*x
        y = y.real
        return y
    
    def generate_init_data(self, initial_size):
        features = np.random.randn(initial_size, self.feature_dim) 
        targets = [float(self.evaluate(f) + np.random.normal(0,0.2,1)) for f in features]
        targets = np.array(targets, dtype='float32')
        return features, targets
    
    def generate_plot_data(self, data_length):
        features = np.linspace(-4, 4, data_length)
        targets = [self.evaluate(f) for f in features]
        targets = np.array(targets, dtype='float32')
        return features, targets


class MyMLP(nn.Module):
    '''A three-layer MLP as classifier'''
    def __init__(self, input_dim, output_dim):
        super(MyMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.output_dim) # output prob 
        self.sm = nn.Sigmoid()
        
    def forward(self, din):
        # print("input din.shape: {}".format(din.shape))
        din = din.view(-1, self.input_dim) # (num_sample, 784)
        dout = F.relu(self.fc1(din)) # 
        dout = F.relu(self.fc2(dout))
        inferences = F.softmax(self.fc3(dout))[:,1]
        regs = 0.0
        return inferences, regs
    
    def compute_loss(self, inferences, labels, regs=0):
        # labels = torch.reshape(labels, [-1, 1])
        # loss = F.mse_loss(inferences, labels)
        # loss_calc = nn.BCELoss()
        loss = F.binary_cross_entropy(inferences, labels)
        # loss_calc = inferences, labels
        return loss + regs

    def compute_loss_neg(self, inferences, labels, regs=0):
        loss = F.binary_cross_entropy(inferences, labels)
        return - loss - regs

    def compute_accuracy(self, inferences, labels):
        inferences_to_label = inferences > 0.5 # logistic's method
        # print(inferences_to_label == labels)
        accuracy = sum(inferences_to_label == labels)/int(labels.shape[0])
        return accuracy

class MyRandomForest(nn.Module):
    '''A three-layer MLP as classifier'''
    def __init__(self, input_dim, output_dim):
        super(MyRandomForest, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.output_dim) # output prob 
        self.sm = nn.Sigmoid()
        
    def forward(self, din):
        # print("input din.shape: {}".format(din.shape))
        din = din.view(-1, self.input_dim) # (num_sample, 784)
        dout = F.relu(self.fc1(din)) # 
        dout = F.relu(self.fc2(dout))
        inferences = F.softmax(self.fc3(dout))[:,1]
        regs = 0.0
        return inferences, regs
    
    def compute_loss(self, inferences, labels, regs=0):
        # labels = torch.reshape(labels, [-1, 1])
        # loss = F.mse_loss(inferences, labels)
        # loss_calc = nn.BCELoss()
        loss = F.binary_cross_entropy(inferences, labels)
        # loss_calc = inferences, labels
        return loss + regs

    def compute_loss_neg(self, inferences, labels, regs=0):
        loss = F.binary_cross_entropy(inferences, labels)
        return - loss - regs

    def compute_accuracy(self, inferences, labels):
        inferences_to_label = inferences > 0.5 # logistic's method
        # print(inferences_to_label == labels)
        accuracy = sum(inferences_to_label == labels)/int(labels.shape[0])
        return accuracy


def model_argmax(model, features, steps=4000):
    # inferences, regs = self.forward(features)
    x_optim = torch.tensor([0.0], requires_grad=True)
    prob_optimizer = torch.optim.LBFGS([x_optim], lr=0.5) # default hp
    # prob_optimizer = torch.optim.Adam([x_optim], lr=0.05)
    for step in range(steps):
        def closure():
            prob_optimizer.zero_grad()
            pred, regs = model(x_optim)
            # pred = -pred # argmax pred
            loss = -pred
            loss.backward()
            return loss
        prob_optimizer.step(closure)
        if step % 2000 == 0:
            pred, regs = model(x_optim)
            print ('optim step {}: x = {}, f(x) = {}'.format(step, x_optim.tolist(), pred.item()))
    return x_optim



def create_data_density(features, targets, train_portion=0.8, q=0.25):
    data_size = features.shape[0]
    tau = np.quantile(targets, q)
    # print("q: {}, tau: {}".format(q, tau))
    labels = np.less(targets, tau) # y<=tau => 1; y>tau => 0 
    labels = np.array(labels, dtype='int')
    # features = features.to(torch.float32)
    # labels = labels.to(torch.float32)
    train_size = int(data_size*train_portion)
    train_queue = [torch.tensor(features[:train_size], dtype=torch.float32), 
                   torch.tensor(targets[:train_size], dtype=torch.float32), 
                   torch.tensor(labels[:train_size], dtype=torch.float32)]
    test_queue = [torch.tensor(features[train_size:], dtype=torch.float32), 
                   torch.tensor(targets[train_size:], dtype=torch.float32), 
                   torch.tensor(labels[train_size:], dtype=torch.float32)]
    return train_queue, test_queue # tensor float32 style
'''
  test_queue = [torch.tensor(features[train_size:], dtype=torch.float32),
bore_tmp.py:117: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  torch.tensor(targets[train_size:], dtype=torch.float32),
bore_tmp.py:61: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
  inferences = F.softmax(self.fc3(dout))[:,1]
'''
def update_data_density(x_next, y_next, train_queue, test_queue,train_portion=0.8, q=0.25):
    data_size = features.shape[0]
    tau = np.quantile(targets, q)
    # print("q: {}, tau: {}".format(q, tau))
    labels = np.less(targets, tau) # y<=tau => 1; y>tau => 0 
    labels = np.array(labels, dtype='int')
    # features = features.to(torch.float32)
    # labels = labels.to(torch.float32)
    train_size = int(data_size*train_portion)
    train_queue = [torch.tensor(features[:train_size], dtype=torch.float32), 
                   torch.tensor(targets[:train_size], dtype=torch.float32), 
                   torch.tensor(labels[:train_size], dtype=torch.float32)]
    test_queue = [torch.tensor(features[train_size:], dtype=torch.float32), 
                   torch.tensor(targets[train_size:], dtype=torch.float32), 
                   torch.tensor(labels[train_size:], dtype=torch.float32)]
    return train_queue, test_queue # tensor float32 style 


# train the model
def train_classifier(model, train_queue, optimizer, use_gpu=False): 
    '''train the model for an epoch'''
    train_features, train_targets, labels =  train_queue
    # train_features = torch.from_numpy(train_features).to(torch.float32)
    # print("train_features.shape: {} in train classifier\n".format(train_features.shape))
    # train_features = train_features
    labels = labels.to(torch.float32)
    model.train()
    optimizer.zero_grad()
    model.zero_grad()
    inferences, regs = model(train_features)
    # print(inferences.shape)
    # loss = F.binary_cross_entropy(inferences, labels)
    loss = model.compute_loss(inferences, labels, regs=0)
    # loss = loss.to(torch.float32) # modified yan
    loss.backward()
    optimizer.step()
    if use_gpu:
        return loss.cpu().detach().numpy().tolist()
    else:
        return loss.detach().numpy().tolist()

def evaluate_classifier(model, test_queue, use_gpu=False):
    '''evaluate at the end of epoch'''
    model.eval()
    test_features, test_targets, labels = test_queue
    inferences, reg = model(test_features)
    loss = F.binary_cross_entropy(inferences, labels)
    # rmse = torch.sqrt(mse)
    inferences_to_label = inferences > 0.5
    # print(inferences_to_label == labels)
    accuracy = sum(inferences_to_label == labels)/int(labels.shape[0])
    if use_gpu:
        return accuracy.cpu().detach().numpy().tolist()
    else:
        return accuracy.detach().numpy().tolist()


if __name__ == '__main__':
    # build model of an nlp keras(tf1 env)
    feature_dim = 1
    blackbox = FuncBlackBox(feature_dim)
    # initial design
    features, targets = blackbox.generate_init_data(initial_size=40)
    # print("features: {}, targets: {}".format(features, targets))
    print("features.shape: {}, targets.shape: {}".format(features.shape, targets.shape))
    features_plot, targets_plot = blackbox.generate_plot_data(data_length=1000) # just for plot
    plot_figure = True
    if plot_figure:
        plt.ion()
        plt.scatter(features, targets, marker='x')
        plt.plot(features_plot, targets_plot, c='g')
        # plt.show()
        plt.pause(5) 
        plt.close()
    
    features = torch.tensor(features)
    targets = torch.tensor(targets)
    classifier = MyMLP(input_dim=feature_dim, output_dim=2)
    for param in classifier.parameters():
        init.normal_(param, mean=0, std=0.01)
    
    
    losses = []
    torch.manual_seed(1)
    optimizer = torch.optim.SGD(classifier.parameters() , lr=0.01, momentum=0.9)
    train_queue, test_queue = create_data_density(features, targets, train_portion=0.8, q=0.25)
    # print(train_features, train_targets, labels)
    # print(train_queue[0].shape)
    # loss = train_classifier(classifier, train_queue, optimizer, use_gpu=False)
    # score = evaluate_classifier(classifier, test_queue, use_gpu=False)
    
    train_epochs = 2000
    for train_epoch in range(train_epochs):
        # enumerate mini batches
        # train_model(train_dl, model, labels_train, optimizer, criteration)
        loss = train_classifier(classifier, train_queue, optimizer, use_gpu=False)
        losses.append(loss)
        with torch.no_grad():
            score = evaluate_classifier(classifier, test_queue, use_gpu=False)
        print("Epoch[{}|{}], loss: {:.4f}, eval score: {:.5f}".format(train_epoch+1, train_epochs, loss, score))
        x_next = model_argmax(classifier, features, steps=2000+1)
        y_next = blackbox.evaluate(x_next)
        # print('type(features): {}, type(targets): {}, type(x_next): {}, type(y_next): {}'.format(type(features), type(targets), type(x_next), type(y_next)))
        # x_next = torch.tensor(x_next)
        # y_next = torch.tensor(y_next)
        # # update dataset
        # print('features.shape: {}, x_next.shape: {}'.format(features.shape, x_next.shape))
        # x_next.view(1,1)
        features = features.tolist()
        features.append([x_next.item()])
        features = torch.tensor(features)
        
        targets = targets.tolist()
        targets.append(y_next.item())
        targets = torch.tensor(targets)
        # targets = torch.cat(targets, y_next)
        # update model weights
    
        # # suggest new candidate
        # x_next = classifier.argmax(method="L-BFGS-B", num_start_points=3, bounds=0.2)

        # # evaluate blackbox function
        # y_next = blackbox.evaluate(x_next) # blackbox our model, like 

        # # update dataset
        # features = np.concatenate(features, x_next)
        # targets = np.concatenate(targets, y_next)