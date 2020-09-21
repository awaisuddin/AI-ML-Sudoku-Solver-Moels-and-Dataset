# imports
import numpy as np
import pandas as pd

from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Input
from keras.utils import to_categorical


def load_data(nb_train=40000, nb_test=10000, full=False):
    """
    Function to load data in the keras way.
    
    Parameters
    ----------
    nb_train (int): Number of training examples
    nb_test (int): Number of testing examples
    full (bool): If True, whole csv will be loaded, nb_test will be ignored
    
    Returns
    -------
    Xtrain, ytrain (np.array, np.array),
        shapes (nb_train, 9, 9), (nb_train, 9, 9): Training samples
    Xtest, ytest (np.array, np.array), 
        shapes (nb_test, 9, 9), (nb_test, 9, 9): Testing samples
    """
    # if full is true, load the whole dataset
    if full:
        sudokus = pd.read_csv('./sudoku.csv').values
    else:
        sudokus = next(
            pd.read_csv('./sudoku.csv', chunksize=(nb_train + nb_test))
        ).values
        
    quizzes, solutions = sudokus.T
    flatX = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9))
                      for flatten_grid in quizzes])
    flaty = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9))
                      for flatten_grid in solutions])
    
    return (flatX[:nb_train], flaty[:nb_train]), (flatX[nb_train:], flaty[nb_train:])


def diff(grids_true, grids_pred):
    """
    This function shows how well predicted quizzes fit to actual solutions.
    It will store sum of differences for each pair (solution, guess)
    
    Parameters
    ----------
    grids_true (np.array), shape (?, 9, 9): Real solutions to guess in the digit repesentation
    grids_pred (np.array), shape (?, 9, 9): Guesses
    
    Returns
    -------
    diff (np.array), shape (?,): Number of differences for each pair (solution, guess)
    """
    return (grids_true != grids_pred).sum((1, 2))


def delete_digits(X, n_delet=1):
    """
    This function is used to create sudoku quizzes from solutions
    
    Parameters
    ----------
    X (np.array), shape (?, 9, 9, 9|10): input solutions grids.
    n_delet (integer): max number of digit to suppress from original solutions
    
    Returns
    -------
    grids: np.array of grids to guess in one-hot way. Shape: (?, 9, 9, 10)
    """
    grids = X.argmax(3)  # get the grid in a (9, 9) integer shape
    for grid in grids:
        grid.flat[np.random.randint(0, 81, n_delet)] = 0  # generate blanks (replace = True)
        
    return to_categorical(grids)


def batch_smart_solve(grids, solver):
    """
    NOTE : This function is ugly, feel free to optimize the code ...
    
    This function solves quizzes in the "smart" way. 
    It will fill blanks one after the other. Each time a digit is filled, 
    the new grid will be fed again to the solver to predict the next digit. 
    Again and again, until there is no more blank
    
    Parameters
    ----------
    grids (np.array), shape (?, 9, 9): Batch of quizzes to solve (smartly ;))
    solver (keras.model): The neural net solver
    
    Returns
    -------
    grids (np.array), shape (?, 9, 9): Smartly solved quizzes.
    """
    grids = grids.copy()
    for _ in range((grids == 0).sum((1, 2)).max()):
        preds = np.array(solver.predict(to_categorical(grids)))  # get predictions
        probs = preds.max(2).T  # get highest probability for each 81 digit to predict
        values = preds.argmax(2).T + 1  # get corresponding values
        zeros = (grids == 0).reshape((grids.shape[0], 81))  # get blank positions

        for grid, prob, value, zero in zip(grids, probs, values, zeros):
            if any(zero):  # don't try to fill already completed grid
                where = np.where(zero)[0]  # focus on blanks only
                confidence_position = where[prob[zero].argmax()]  # best score FOR A ZERO VALUE (confident blank)
                confidence_value = value[confidence_position]  # get corresponding value
                grid.flat[confidence_position] = confidence_value  # fill digit inplace
    return grids

print("All preprocessing functions has been defined and also has been imported properly")



input_shape = (9, 9, 10)
(_, ytrain), (Xtest, ytest) = load_data()  # We won't use _. We will work directly with ytrain

# one-hot-encoding --> shapes become :
# (?, 9, 9, 10) for Xs
# (?, 9, 9, 9) for ys
Xtrain = to_categorical(ytrain).astype('float32')  # from ytrain cause we will creates quizzes from solusions
Xtest = to_categorical(Xtest).astype('float32')

ytrain = to_categorical(ytrain-1).astype('float32') # (y - 1) because we 
ytest = to_categorical(ytest-1).astype('float32')   # don't want to predict zeros


print("Now , you have Xtrain , Xtest , ytrain , ytest ready to use")

