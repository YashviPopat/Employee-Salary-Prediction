
import pandas as pd
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import tensor


from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


# In[3]:


def data_preprocessing(task_1a_dataframe):
    '''
    Purpose:
    ---
    This function will be used to load your csv dataset and preprocess it.
    Preprocessing involves cleaning the dataset by removing unwanted features,
    decision about what needs to be done with missing values etc. Note that
    there are features in the csv file whose values are textual (eg: Industry,
    Education Level etc)These features might be required for training the model
    but can not be given directly as strings for training. Hence this function
    should return encoded dataframe in which all the textual features are
    numerically labeled.

    Input Arguments:
    ---
    `task_1a_dataframe`: [Dataframe]
                          Pandas dataframe read from the provided dataset

    Returns:
    ---
    `encoded_dataframe` : [ Dataframe ]
                          Pandas dataframe that has all the features mapped to
                          numbers starting from zero

    Example call:
    ---
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    '''

    #################	ADD YOUR CODE HERE	##################
    df = task_1a_dataframe
    df['Education'] = df['Education'].astype('category')
    encode_map = {
        'Masters': 0,
        'Bachelors': 1,
        'PHD':2    
    }
    df['Education'].replace(encode_map, inplace=True)

    df['City'] = df['City'].astype('category')
    encode_map = {
        'Bangalore': 0,
        'New Delhi': 1,
        'Pune':2    
    }
    df['City'].replace(encode_map, inplace=True)

    df['Gender'] = df['Gender'].astype('category')
    encode_map = {
        'Male': 0,
        'Female': 1,    
    }
    df['Gender'].replace(encode_map, inplace=True)

    df['EverBenched'] = df['EverBenched'].astype('category')
    encode_map = {
        'No': 0,
        'Yes': 1,
    }
    df['EverBenched'].replace(encode_map, inplace=True)
    encoded_dataframe=df
    ##########################################################
    return encoded_dataframe




# In[4]:


def identify_features_and_targets(encoded_dataframe):
    '''
    Purpose:
    ---
    The purpose of this function is to define the features and
    the required target labels. The function returns a python list
    in which the first item is the selected features and second
    item is the target label

    Input Arguments:
    ---
    `encoded_dataframe` : [ Dataframe ]
                        Pandas dataframe that has all the features mapped to
                        numbers starting from zero

    Returns:
    ---
    `features_and_targets` : [ list ]
                            python list in which the first item is the
                            selected features and second item is the target label

    Example call:
    ---
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    '''

    #################	ADD YOUR CODE HERE	##################
    X = encoded_dataframe.iloc[:, 0:-1]
    y = encoded_dataframe.iloc[:, -1]
    features_and_targets=[X,y]
    ##########################################################

    return features_and_targets


# In[5]:


def load_as_tensors(features_and_targets):

    '''
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training
    and validation, and then load them as as tensors.
    Training of the model requires iterating over the training tensors.
    Hence the training sensors need to be converted to iterable dataset
    object.

    Input Arguments:
    ---
    `features_and targets` : [ list ]
                            python list in which the first item is the
                            selected features and second item is the target label

    Returns:
    ---
    `tensors_and_iterable_training_data` : [ list ]
                                            Items:
                                            [0]: X_train_tensor: Training features loaded into Pytorch array
                                            [1]: X_test_tensor: Feature tensors in validation data
                                            [2]: y_train_tensor: Training labels as Pytorch tensor
                                            [3]: y_test_tensor: Target labels as tensor in validation data
                                            [4]: Iterable dataset object and iterating over it in
                                                 batches, which are then fed into the model for processing

    Example call:
    ---
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    '''

    #################	ADD YOUR CODE HERE	##################
    X = features_and_targets[0]
    y = features_and_targets[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Convert the features and labels into PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)  # Convert to float32 if not already
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Create a custom dataset object
    class CustomDataset(Dataset):
        def __init__(self, data, target):
            self.data = data
            self.target = target

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.target[idx]
    
    # Create an iterable dataset object for training
    dataset = CustomDataset(X_train_tensor, y_train_tensor)

    # Create a DataLoader object for iterating over the dataset in batches
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    tensors_and_iterable_training_data = [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, data_loader]
    ###########################################################################
    return tensors_and_iterable_training_data


# In[6]:


class Salary_Predictor(nn.Module):
    '''
    Purpose:
    ---
    The architecture and behavior of your neural network model will be
    defined within this class that inherits from nn.Module. Here you
    also need to specify how the input data is processed through the layers.
    It defines the sequence of operations that transform the input data into
    the predicted output. When an instance of this class is created and data
    is passed through it, the `forward` method is automatically called, and
    the output is the prediction of the model based on the input data.

    Returns:
    ---
    `predicted_output` : Predicted output for the given input data
    '''
    def __init__(self):
        super(Salary_Predictor, self).__init__()
        '''
        Define the type and number of layers
        '''
        #######	ADD YOUR CODE HERE	#######
        self.layer_1 = nn.Linear(8, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64,64)
        self.layer_out = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(64)
        ###################################

    def forward(self, x):
        '''
        Define the activation functions
        '''
        #######	ADD YOUR CODE HERE	#######
        predicted_output= self.relu(self.layer_1(x))
        predicted_output = self.batchnorm1(predicted_output)
        predicted_output = self.relu(self.layer_2(predicted_output))
        predicted_output = self.batchnorm2(predicted_output)
        predicted_output = self.dropout(predicted_output)
        predicted_output = self.layer_out(predicted_output)
        ###################################

        return predicted_output


# In[7]:


def model_loss_function():
    '''
    Purpose:
    ---
    To define the loss function for the model. Loss function measures
    how well the predictions of a model match the actual target values
    in training data.

    Input Arguments:
    ---
    None

    Returns:
    ---
    `loss_function`: This can be a pre-defined loss function in PyTorch
                    or can be user-defined

    Example call:
    ---
    loss_function = model_loss_function()
    '''
    #################	ADD YOUR CODE HERE	##################
    loss_function = nn.BCEWithLogitsLoss()
    ##########################################################

    return loss_function


# In[8]:


def model_optimizer(model):
    '''
    Purpose:
    ---
    To define the optimizer for the model. Optimizer is responsible
    for updating the parameters (weights and biases) in a way that
    minimizes the loss function.

    Input Arguments:
    ---
    `model`: An object of the 'Salary_Predictor' class

    Returns:
    ---
    `optimizer`: Pre-defined optimizer from Pytorch

    Example call:
    ---
    optimizer = model_optimizer(model)
    '''
    #################	ADD YOUR CODE HERE	##################
    LEARNING_RATE= 0.001
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    ##########################################################

    return optimizer


# In[13]:


def model_number_of_epochs():
    '''
    Purpose:
    ---
    To define the number of epochs for training the model

    Input Arguments:
    ---
    None

    Returns:
    ---
    `number_of_epochs`: [integer value]

    Example call:
    ---
    number_of_epochs = model_number_of_epochs()
    '''
    #################	ADD YOUR CODE HERE	##################
    number_of_epochs = 200
    ##########################################################

    return number_of_epochs



# In[14]:


def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    '''
    Purpose:
    ---
    All the required parameters for training are passed to this function.

    Input Arguments:
    ---
    1. `model`: An object of the 'Salary_Predictor' class
    2. `number_of_epochs`: For training the model
    3. `tensors_and_iterable_training_data`: list containing training and validation data tensors
                                             and iterable dataset object of training tensors
    4. `loss_function`: Loss function defined for the model
    5. `optimizer`: Optimizer defined for the model

    Returns:
    ---
    trained_model

    Example call:
    ---
    trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

    '''
    #################	ADD YOUR CODE HERE	##################
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterable_training_data = tensors_and_iterable_training_data
    
    model.train()
    for epoch in range(number_of_epochs):
        for batch in iterable_training_data:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.unsqueeze(1)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{number_of_epochs}')
        print(loss)
    trained_model=model.eval()
    ##########################################################

    return trained_model



# In[15]:


def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    ---
    This function will utilise the trained model to do predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.

    Input Arguments:
    ---
    1. `trained_model`: Returned from the training function
    2. `tensors_and_iterable_training_data`: list containing training and validation data tensors
                                             and iterable dataset object of training tensors

    Returns:
    ---
    model_accuracy: Accuracy on the validation dataset

    Example call:
    ---
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

    '''
    #################	ADD YOUR CODE HERE	##################
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterable_training_data = tensors_and_iterable_training_data
    y = []
    class TestData(Dataset):
    
        def __init__(self, X_data):
            self.X_data = X_data

        def __getitem__(self, index):
            return self.X_data[index]

        def __len__ (self):
            return len(self.X_data)
    test_data = TestData(torch.FloatTensor(X_test_tensor))
    X_test_tensor = X_test_tensor.numpy()
    test_data = TestData(X_test_tensor)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trained_model.eval()
    with torch.no_grad():
    # Get the predictions
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y.append(y_pred_tag.cpu().numpy())

    y = [a.squeeze().tolist() for a in y]
    model_accuracy=classification_report(y_test_tensor, y)
    ##########################################################

    return model_accuracy


if __name__ == "__main__":

    # reading the provided dataset csv file using pandas library and
    # converting it to a pandas Dataframe
    task_1a_dataframe = pd.read_csv('task_1a_dataset.csv')

    # data preprocessing and obtaining encoded data
    encoded_dataframe = data_preprocessing(task_1a_dataframe)

    # selecting required features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)

    # obtaining training and validation data tensors and the iterable
    # training data object
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

    # model is an instance of the class that defines the architecture of the model
    model = Salary_Predictor()

    # obtaining loss function, optimizer and the number of training epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

    # training the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data,
                    loss_function, optimizer)

    # validating and obtaining accuracy
    model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")

    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")