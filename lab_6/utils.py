import sys
import numpy as np
import matplotlib.pyplot as plt


# Split the data into training and validation split
def train_validation_split(inputs, targets, tr_split=0.5):
    """ This function splits the training data and validation data, based on the training
    testing split
    """
    num_train_samples = int(tr_split * len(inputs))
    num_val_samples = len(inputs) - num_train_samples
    tr_inputs = torch.Tensor(inputs[:num_train_samples, :, :])
    tr_targets = torch.Tensor(targets[:num_train_samples])
    val_inputs = torch.Tensor(inputs[num_train_samples:, :, :])
    val_targets = torch.Tensor(targets[num_train_samples:])

    return tr_inputs, tr_targets, val_inputs, val_targets

# Plot the losses for training and validation set
def plot_losses(tr_losses, val_losses, logscale=True):
    """ This function plots the training and the validation 
    losses, with an option to plot the error in 'logscale' or
    'linear' scale
    """
    plt.figure(figsize=(10,5))
    if logscale == False:

        plt.plot(tr_losses, 'r+-')
        plt.plot(val_losses, 'b*-')
        plt.xlabel("No. of training iterations", fontsize=16)
        plt.ylabel("MSE Loss", fontsize=16)
        plt.legend(['Training Set', 'Validation Set'], fontsize=16)
        plt.title("MSE loss vs. no. of training iterations", fontsize=20)

    elif logscale == True:

        plt.plot(np.log10(tr_losses), 'r+-')
        plt.plot(np.log10(val_losses), 'b*-')
        plt.xlabel("No. of training iterations", fontsize=16)
        plt.ylabel("Log of MSE Loss", fontsize=16)
        plt.legend(['Training Set', 'Validation Set'], fontsize=16)
        plt.title("Log of MSE loss vs. no. of training iterations", fontsize=20)


# Plotting the predictions for the testing cycle
def plot_predictions(ytest, predictions, title):
    """ This function plots the predictions versus the true signal in 
    a single cycle
    """
    plt.figure(figsize=(10,5))
    #Prediction plot
    #plt.title("Prediction value of number of sunspots vs time index", fontsize=20)
    plt.title(title, fontsize=20)
    plt.plot(ytest[:,0], ytest[:,1], '+-', label="actual test signal", color="orange")
    plt.plot(ytest[:,0], predictions, '*-', label="prediction", color="green")
    plt.legend(fontsize=16)
    plt.show()

# Plotting the future predictions 
def plot_future_predictions(data, minimum_idx, ytrain, predictions, title=None):
    """ This function plots the future predictions for a single cycle (roughly)
    along with the existing dataset
    """
    plt.figure(figsize=(10,5))
    resolution = np.around(np.diff(data[:,0]).mean(),1)
    plt.plot(data[:minimum_idx[-1],0], data[:minimum_idx[-1],1], 'r+-')
    plt.plot(np.arange(ytrain[-1][-1][0] + resolution, ((len(predictions)) * resolution) + 
        ytrain[-1][-1][0], resolution), predictions, 'b*-')
    plt.legend(['Original timeseries', 'Future prediction'], fontsize=16)
    if title is None:
        plt.title('Plot of original timeseries and future predictions', fontsize=20)
    else:
        plt.title(title, fontsize=20)
    plt.show()



def get_msah_training_dataset(X, minimum_idx, tau=1, p=np.inf):
    """ This function returns the data and targets on a cycle-wise basis, based
    on the indices of the starting and ending points of cycles, number of points to
    predict ahead, and number of points used for predicting each sample.
    
    Args:
        X:           (n_samples,2), time series with timestamps on the first column and values on the second
        minimum_idx: The index on the minimums 
        tau:         Number of the steps ahead (to predict)
        p:           Order of model (number of steps backwards to use in the training data),
    Returns:
        xtrain: List of lists,
          the ith element of `xtrain` is a list containing the training data relative to the prediction 
          of the samples in the ith cycle. if p is np.inf, The training data consists in the data up 
          to the start of the ith cycle
        Y: List of np.ndarray, the ith element of `Y` is the raw data of the ith cycle 

    """
    Y = []
    xtrain = []

    for idx in range(1, minimum_idx.shape[0]):
        tmp = []
        if not np.isinf(p):
            # i spans the indexes of a cycle
            for i in range(minimum_idx[idx - 1], minimum_idx[idx]):
                # tmp receives a tuple (training data, target)
                if i - p >= 0:
                    # Append the p points prior to i
                    tmp.append((X[i - p:i, :], X[i:i + tau]))
                else:
                    # Do not append the data segment if it is shorter than p
                    pass

            #tmp.append((X[0:i, :], X[i:i + tau]))
            xtrain.append(tmp)
        else:
            # If p is given as np.inf, in that case the entire signal is used for
            # prediction relative to the target
            xtrain.append(X[:minimum_idx[idx]])

        if idx + 1 < minimum_idx.shape[0]:
            Y.append(X[minimum_idx[idx]:minimum_idx[idx + 1], :])

    return xtrain, Y

def concat_data(x, col=1):
    """ Concatenate all the `col` column of the element
    """
    if col == 1:
        return np.concatenate([xx[:, col].reshape(1, -1) for xx in x], axis=0)
    elif col == -1:
        return np.concatenate([xx[:, :].reshape(1, -1) for xx in x], axis=0)

def get_cycle(X, Y, idx):
    """ This function gets the training data, training targets and test
    targets using the set of data, targets and cycle index
    """
    if isinstance(X[0], np.ndarray):
        xtrain = X[idx]
        ytrain = None
    else:
        tmp = sum(X[:idx + 1], [])
        xtrain = [t[0] for t in tmp]
        ytrain = [t[1] for t in tmp]

    if idx == len(X) - 1:
        ytest = np.array([])
    else:
        ytest = Y[idx]
    return xtrain, ytrain, ytest

def normalize(X, feature_space=(0, 1)):
    """ Normalizing the features in the feature_space (lower_lim, upper_lim)

    Args:
        X ([numpy.ndarray]): Unnormalized data consisting of signal points
        feature_space (tuple, optional): [lower and upper limits]. Defaults to (0, 1).

    Returns:
        X_norm [numpy.ndarray]: Normalized feature values
    """
    X_norm = (X - X.min())/(X.max() - X.min()) * (feature_space[1] - feature_space[0]) + \
        feature_space[0]
    return X_norm, X.max(), X.min()

def unnormalize(X_norm, X_max, X_min, feature_space=(0, 1)):
    """ Normalizing the features in the feature_space (lower_lim, upper_lim)

    Args:
        X_norm ([numpy.ndarray]): Unnormalized data consisting of signal points
        X_max, X_min: Maximum and minimum values prior to normalization
        feature_space (tuple, optional): [lower and upper limits]. Defaults to (0, 1).

    Returns:
        X_norm [numpy.ndarray]: Normalized feature values
    """
    X_unnorm = ((X_norm - feature_space[0]) / (feature_space[1] - feature_space[0])) * (X_max - X_min) + \
        X_min
    return X_unnorm

def get_minimum(X, dataset):
    """ This function returns the 'minimum' indices or the indices for
    the solar cycles for the particular dataset type.

    Args:
        X ([numpy.ndarray]): The complete time-series data present as (N_samples x 2),
        with each row being of the form (time-stamp x signal value)
        dataset ([str]): String to indicate the type of the dataset - solar / dynamo

    Returns:
        minimum_idx : An array containing the list of indices for the minimum 
        points of the data
    """
    if dataset == "dynamo":
        minimum_idx = find_index_of_minimums_dyn(X[:, 1])
    else:
        print("Dataset {} unknown".format(dataset))
        sys.exit(1)
    return minimum_idx

def find_index_of_minimums_dyn(dynamo_signal):
    """ This function finds the indices of the solar cycles for the 'dynamo' signal
    """
    index_of_minimums = []
    for i in range(1, dynamo_signal.size): # point 0 has not a preceeding point
        is_minimum = check_if_is_minimum(dynamo_signal, i)
        if is_minimum:
            index_of_minimums.append(i)
    return np.array(index_of_minimums).astype(int).reshape(-1)

def check_if_is_minimum(signal, index):
    """ This auxillary function tests whether a given point is a minimum point or not
    """
    if signal[index-1] > signal[index] and signal[index+1] > signal[index]:
        is_minium = True
    else:
        is_minium = False
    return is_minium
