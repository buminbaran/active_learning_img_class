import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm



class TrainingClass:
   
   
   def  __init__ (self, model, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
     
     
     
   def train(self, dataloader, lr=0.001, epochs=10):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        epoch_losses = [] 

        epoch_loop = tqdm(range(epochs), desc="Training Progress")
        for epoch in epoch_loop:
            self.model.train()
            running_loss = 0.0
            
            batch_loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)#tqdm for progress bar like in ml lab
            
            for inputs, labels in batch_loop:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(inputs) # forward pass
                loss = criterion(outputs, labels)

               
                loss.backward() # backward pass
                optimizer.step() 

                running_loss += loss.item()

                batch_loop.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(dataloader)
            epoch_losses.append(avg_loss) 
            
            epoch_loop.set_postfix(avg_loss=f"{avg_loss:.4f}")
            
        return epoch_losses

   
   def evaluate(self, dataloader):

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():  # no need to calc gradients for eval
            for inputs, labels in dataloader:

                inputs = inputs.to(self.device)
                labels =  labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy on the test set: {accuracy:.2f} %')
        return accuracy

   def predict_probas(self, dataloader):

        self.model.eval() 
        all_probas = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                # softmax to get probabilities
                probas = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probas.append(probas)

        return np.vstack(all_probas)