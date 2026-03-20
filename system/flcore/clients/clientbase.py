"""
Base client for VPFL
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from utils.data_utils import get_dataloader


class Client(object):
    """
    Base client for federated learning
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.args = args
        self.id = id
        self.device = args.device
        self.model = copy.deepcopy(args.model)
        
        self.train_samples = train_samples
        self.test_samples = test_samples
        
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0}
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=args.momentum if hasattr(args, 'momentum') else 0
        )
        
        self.learning_rate_decay = args.learning_rate_decay if hasattr(args, 'learning_rate_decay') else False
        if self.learning_rate_decay:
            self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, 
                gamma=args.learning_rate_decay_gamma if hasattr(args, 'learning_rate_decay_gamma') else 0.99
            )
    
    def load_train_data(self, batch_size=None):
        """Load training data"""
        if batch_size is None:
            batch_size = self.batch_size
        
        from utils.data_utils import read_client_data
        train_data = read_client_data(self.args.dataset, self.id, is_train=True)
        return get_dataloader(train_data, batch_size, shuffle=True)
    
    def load_test_data(self, batch_size=None):
        """Load test data"""
        if batch_size is None:
            batch_size = self.batch_size
        
        from utils.data_utils import read_client_data
        test_data = read_client_data(self.args.dataset, self.id, is_train=False)
        return get_dataloader(test_data, batch_size, shuffle=False)
    
    def set_parameters(self, model):
        """Set model parameters from global model"""
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
    
    def clone_model(self, model, target):
        """Clone model parameters"""
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
    
    def test_metrics(self):
        """Test model and return metrics"""
        testloader = self.load_test_data()
        self.model.eval()
        
        test_acc = 0
        test_loss = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                test_loss += loss.item() * y.shape[0]
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        
        return test_acc, test_loss, test_num
    
    def train_metrics(self):
        """Train and return metrics"""
        trainloader = self.load_train_data()
        self.model.train()
        
        train_acc = 0
        train_loss = 0
        train_num = 0
        
        for x, y in trainloader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * y.shape[0]
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            train_num += y.shape[0]
        
        return train_acc, train_loss, train_num
