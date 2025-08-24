import csv
import os
import torch
from torch import nn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/census-bureau-scaled.csv")

features = [col for col in df.columns if col != 'label' and col != "Unnamed: 0"]
target_column = "label"

# MSE_Loss, pytorch has its own implementation, but I made it too in order to make sure there's no bug and for testing
class MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        # Implementation of custom loss calculation
        diff = torch.sub(y_true, y_pred)
        return torch.mean(torch.square(diff))

class neural_network(nn.Module):
    # we initialize the neural net with a bunch of embedding layers and the actual neural network
    def __init__(self, n = 1000):
        super().__init__()
        # could explore built-in embedding as an alternative
        '''self.embeddings = []

        # embedding layers for the categorical variables
        count = 0
        for feature in features:
            n_categories = df[feature].nunique()
            embed_dim = n_categories#min(50, (n_categories + 1) // 2)
            count+=embed_dim
            self.embeddings.append(nn.Embedding(n_categories, embed_dim))'''
        

        
        # the actual neural network - for binary classification with BCELoss
        self.model = nn.Sequential(
            nn.Linear(41, n),
            nn.Sigmoid(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, 1),  # Single output for binary classification
            nn.Sigmoid()       # Sigmoid to get probability between 0 and 1
        )
        
    # predicting the labels for the data in x
    def forward(self, x):
        '''embed_inputs = []
        for feature_index in range(len(features)):
            embed = self.embeddings[feature_index](x[:,feature_index])
            embed_inputs.append(embed)
        
        x = torch.cat(embed_inputs, dim = 1)'''
        #print(x[0,:10])
        #print(x.shape)
        return self.model(x)


# takes in dataframe for data
def train(model, optimizer, data, loss_funct,         epochs=5,      # ↓ fewer passes
          batches=50,    # ↓ fewer mini‐batches per epoch
          batch_size=32,  # ↑ bigger chunks
          verbose = False
         ):

    # each epoch shuffles and goes over the data
    for epoch in range(epochs):
        shuffled = data.sample(frac = 1)
        # Convert features to float (not int!) - this fixes the dtype mismatch error
        x = torch.from_numpy(shuffled[features].values).float()
        # Extract label column values directly to get 1D tensor
        y = torch.from_numpy(shuffled[target_column].values).float()
        running_loss = 0.0

        # i is the batch number for this epoch
        for i in range(min(batches, int(data.shape[0]/batch_size))):#range(data.shape[0]):
            inputs = x[batch_size*i : batch_size*(i+1)]
            labels = y[batch_size*i : batch_size*(i+1)]

            # here is the actual running and stochastic gradient descent
            optimizer.zero_grad()

            outputs = model(inputs)
            
            # BCELoss expects inputs to be squeezed to remove extra dimensions
            # and targets to be the same shape
            outputs_squeezed = outputs.squeeze()
            labels_squeezed = labels.squeeze()
            
            # Ensure both have the same shape for BCELoss
            if outputs_squeezed.dim() == 0:
                outputs_squeezed = outputs_squeezed.unsqueeze(0)
            if labels_squeezed.dim() == 0:
                labels_squeezed = labels_squeezed.unsqueeze(0)
            
            loss = loss_funct(outputs_squeezed, labels_squeezed)
            loss.backward()
            optimizer.step()

            # we print the loss so far
            # bit convoluted, but we print every epoch (can be changed to print more frequently)
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/10 :.5f}')
                if verbose:
                    # Show probabilities and predictions
                    probs = outputs_squeezed[0:5].detach()
                    preds = (probs >= 0.5).long()
                    print("example: ", "predicted probs", probs, ", predictions", preds, ", actual", labels_squeezed[0:5],'\n')
                running_loss = 0.0
        print("------------------------------")

# function that evaluates loss
def evaluate_loss(model, data, loss_funct, verbose = False):
    # Convert features to float (not int!)
    x = torch.from_numpy(data[features].values).float()
    # Extract label column values directly to get 1D tensor
    # BCELoss expects float targets between 0 and 1
    y = torch.from_numpy(data[target_column].values).float()
    pred = model(x)
    
    # Squeeze both tensors to ensure proper shapes for BCELoss
    pred_squeezed = pred.squeeze()
    y_squeezed = y.squeeze()
    
    if verbose:
        print("example:",pred_squeezed[0], y_squeezed[0],'\n')
    return loss_funct(pred_squeezed, y_squeezed)

# function that converts probabilities into predictions to evaluate accuracy
def evaluate_acc(model, data, verbose = False) :
    # Convert features to float (not int!)
    x = torch.from_numpy(data[features].values).float()
    # Extract label column values directly to get 1D tensor
    # BCELoss expects float targets between 0 and 1
    y = torch.from_numpy(data[target_column].values).float()
    
    # Get predictions (already probabilities due to sigmoid in final layer)
    y_hat = model(x)
    
    # Squeeze to remove extra dimensions
    y_hat_squeezed = y_hat.squeeze()
    y_squeezed = y.squeeze()
    
    if verbose:
        print("Raw outputs (probabilities):", y_hat_squeezed[:5])
        print("True labels:", y_squeezed[:5])
    
    # Convert probabilities to binary predictions (threshold at 0.5)
    pred = (y_hat_squeezed >= 0.5).long()  # threshold at 0.5
    
    if verbose:
        print("Binary predictions:", pred[:5])
    
    # Analyze prediction distribution
    pred_0_count = (pred == 0).sum().item()
    pred_1_count = (pred == 1).sum().item()
    total_predictions = pred.shape[0]
    
    pred_0_proportion = pred_0_count / total_predictions
    pred_1_proportion = pred_1_count / total_predictions
    
    print(f"Prediction Distribution:")
    print(f"  Class 0: {pred_0_count} ({pred_0_proportion:.1%})")
    print(f"  Class 1: {pred_1_count} ({pred_1_proportion:.1%})")
    
    # Analyze true label distribution for comparison
    y_long = y_squeezed.long()
    true_0_count = (y_long == 0).sum().item()
    true_1_count = (y_long == 1).sum().item()
    
    true_0_proportion = true_0_count / total_predictions
    true_1_proportion = true_1_count / total_predictions
    
    print(f"True Label Distribution:")
    print(f"  Class 0: {true_0_count} ({true_0_proportion:.1%})")
    print(f"  Class 1: {true_1_count} ({true_1_proportion:.1%})")
    
    # Check for class imbalance issues
    if pred_0_count == 0:
        print("⚠️  WARNING: Network is only predicting class 1!")
    elif pred_1_count == 0:
        print("⚠️  WARNING: Network is only predicting class 0!")
    elif abs(pred_0_proportion - pred_1_proportion) > 0.8:
        print("⚠️  WARNING: Network predictions are heavily imbalanced!")
    
    # Calculate accuracy - convert y back to long for comparison
    accuracy = 1 - torch.sum(torch.abs(pred - y_long)).item() / pred.shape[0]
    
    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy

# function that splits data into training, validation, and testing sets
def split(data, train_ratio = 0.8, validation_ratio = 0.1):
    shuffled = data.sample(frac = 1)
    divider1 = int(train_ratio*shuffled.shape[0])
    divider2 = int((train_ratio+validation_ratio) * shuffled.shape[0])
    train_set = shuffled[:divider1]
    valid_set = shuffled[divider1 : divider2]
    test_set = shuffled[divider2:]
    return train_set, valid_set, test_set

def balance_dataset(data, target_col="label", method="oversample", random_state=42):
    """
    Balance the dataset to have equal representation of both classes.
    
    Args:
        data (pandas.DataFrame): Input dataset
        target_col (str): Name of the target column
        method (str): Either 'oversample' (duplicate minority) or 'undersample' (remove majority)
        random_state (int): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: Balanced dataset
    """
    print(f"Original dataset shape: {data.shape}")
    print(f"Original class distribution:")
    print(data[target_col].value_counts())
    
    # Get class counts
    class_counts = data[target_col].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    
    print(f"\nMinority class: {minority_class} ({minority_count} samples)")
    print(f"Majority class: {majority_class} ({majority_count} samples)")
    
    if method.lower() == "oversample":
        # Duplicate minority class samples to match majority class
        minority_data = data[data[target_col] == minority_class]
        majority_data = data[data[target_col] == majority_class]
        
        # Calculate how many times to repeat minority samples
        repeat_factor = majority_count // minority_count
        remainder = majority_count % minority_count
        
        # Repeat minority samples
        balanced_minority = pd.concat([minority_data] * repeat_factor, ignore_index=True)
        
        # Add remaining samples randomly if there's a remainder
        if remainder > 0:
            remaining_samples = minority_data.sample(n=remainder, random_state=random_state)
            balanced_minority = pd.concat([balanced_minority, remaining_samples], ignore_index=True)
        
        # Combine balanced minority with majority
        balanced_data = pd.concat([balanced_minority, majority_data], ignore_index=True)
        
        print(f"\nOversampling: Duplicated minority class {repeat_factor} times + {remainder} random samples")
        
    elif method.lower() == "undersample":
        # Randomly remove majority class samples to match minority class
        minority_data = data[data[target_col] == minority_class]
        majority_data = data[data[target_col] == majority_class]
        
        # Randomly sample majority class to match minority count
        balanced_majority = majority_data.sample(n=minority_count, random_state=random_state)
        
        # Combine minority with balanced majority
        balanced_data = pd.concat([minority_data, balanced_majority], ignore_index=True)
        
        print(f"\nUndersampling: Randomly selected {minority_count} majority class samples")
        
    else:
        raise ValueError("Method must be either 'oversample' or 'undersample'")
    
    # Shuffle the balanced dataset
    balanced_data = balanced_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\nBalanced dataset shape: {balanced_data.shape}")
    print(f"Balanced class distribution:")
    print(balanced_data[target_col].value_counts())
    
    return balanced_data

def analyze_class_balance(data, target_col="label"):
    """
    Analyze the class balance in a dataset.
    
    Args:
        data (pandas.DataFrame): Input dataset
        target_col (str): Name of the target column
        
    Returns:
        dict: Analysis results
    """
    class_counts = data[target_col].value_counts()
    total_samples = len(data)
    
    analysis = {
        'total_samples': total_samples,
        'class_counts': class_counts.to_dict(),
        'class_proportions': (class_counts / total_samples).to_dict(),
        'imbalance_ratio': class_counts.max() / class_counts.min(),
        'is_balanced': len(class_counts) == 2 and abs(class_counts.iloc[0] - class_counts.iloc[1]) <= 1
    }
    
    print(f"Class Balance Analysis:")
    print(f"  Total samples: {total_samples}")
    print(f"  Class counts: {analysis['class_counts']}")
    print(f"  Class proportions: {analysis['class_proportions']}")
    print(f"  Imbalance ratio: {analysis['imbalance_ratio']:.2f}")
    print(f"  Is balanced: {analysis['is_balanced']}")
    
    if analysis['imbalance_ratio'] > 10:
        print("HEAVY class imbalance detected!")
    elif analysis['imbalance_ratio'] > 3:
        print("Moderate class imbalance detected!")
    else:
        print("Class balance looks good!")
    
    return analysis

if __name__ == "__main__":
    verbose = False
    train_set, valid_set, test_set  = split(df, train_ratio = 0.6, validation_ratio = 0.2)
    train_set.to_csv("data/training.csv")
    valid_set.to_csv("data/validation.csv")
    test_set.to_csv("data/testing.csv")
    
    # Analyze class balance in training data
    print("-" * 60)
    print("CLASS BALANCE ANALYSIS")
    print("-" * 60)
    train_analysis = analyze_class_balance(train_set, target_column)
    
    # Option to balance the training data
    balance_training = True  # Set to False if you don't want to balance
    balance_method = "oversample"  # "oversample" or "undersample"
    
    if balance_training and not train_analysis['is_balanced']:
        print("\n" + "=" * 60)
        print("BALANCING TRAINING DATA")
        print("=" * 60)
        train_set_balanced = balance_dataset(train_set, target_column, balance_method)
        
        # Save balanced training data
        train_set_balanced.to_csv("data/training_balanced.csv")
        print(f"Balanced training data saved to 'data/training_balanced.csv'")
        
        # Use balanced data for training
        train_set = train_set_balanced
    else:
        print(f"\nSkipping data balancing (balance_training={balance_training})")
    
    print("\n" + "-" * 60)
    print("TRAINING NEURAL NETWORK")
    print("-" * 60)

    model = neural_network(n=800)                       # slightly smaller
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # bigger LR
    
    # Test a single forward pass to debug
    '''print("\nTesting single forward pass...")
    test_x = torch.from_numpy(train_set[features].values[:5]).float()
    test_output = model(test_x)
    print(f"Input shape: {test_x.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output values: {test_output}")
    print(f"Output squeezed shape: {test_output.squeeze().shape}")'''
    
    try:
        train( model,
               optimizer,
               train_set,
               nn.BCELoss(),
               epochs=200,      # five passes
               batches=30,    # ~30 mini‐batches/epoch
               batch_size=64  # 64 samples at once
             )
        print("\n" + "-" * 60)
        print("EVALUATION RESULTS")
        print("-" * 60)
        print("Training set performance:")
        train_acc = evaluate_acc(model, train_set, verbose=True)
        print("\nValidation set performance:")
        val_acc = evaluate_acc(model, valid_set, verbose=True)
        print("\nFull dataset performance:")
        full_acc = evaluate_acc(model, df, verbose=True)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
