import numpy as np
import pandas as pd
import copy

def fill_mat(dims, data, col, output_col):
    mat_dim = dims[:-1] + (len(output_col),) #appende (len(output_col),)
    mat = np.zeros(mat_dim)
    if len(dims) == 0:
        value = []
        value = [data[el].values[0] for el in output_col]
        return np.array(value)
    el_columns = np.sort(np.unique(data[col[0]]))#sorted(list(set(data[col[0]])))#order
    for i in range(dims[0]):
        selected = data[data[col[0]] == el_columns[i]]
        mat[i] = fill_mat( dims[1:], selected, col[1:], output_col)
    return mat   
def txt_to_matrix(filename, delimiter, output_col):
    df = pd.read_csv(filename, delimiter = delimiter)
    col_name = df.columns
    for el in output_col:
        col_name = [x for x in col_name if el not  in x]
    col_name = [x for x in col_name if 'Unnamed' not  in x]
    dims = []
    for el in col_name:
        dims.append(len(set(df[el])))
    print(dims)
    mat = np.zeros(dims)
    data = df
    mat = fill_mat(dims, data, col_name, output_col)
    value_col = []
    for el in col_name:
        value_col.append(sorted(list(set(data[el]))))
    return (mat, col_name, value_col)
def closest_indexes(lst, elements):
    # Initialize a list to store the indexes of the closest elements
    closest_indexes = []
    # Iterate over the elements
    for element in elements:
        # Initialize a variable to store the minimum distance so far
        min_distance = float("inf")
        # Initialize a variable to store the index of the closest element so far
        min_index = 0
        # Iterate over the elements in the list
        for i, el in enumerate(lst):
            # Calculate the absolute distance between the element and the input element
            distance = abs(el - element)
            # If the distance is less than the minimum distance so far, update the minimum distance and index
            if distance < min_distance:
                min_distance = distance
                min_index = i
        # Add the index of the closest element to the list
        closest_indexes.append(min_index)
    # Return the list of indexes
    return closest_indexes


if __name__ == "__main__":
    (mat, col, value_col) = txt_to_matrix("res.txt", "\t", ["VarMua0Opt","VarMus0Opt"])
    import matplotlib.pyplot as plt
    for i in range(24):
        plt.cla()
        plt.plot(mat[:,i,0,1])
        plt.ylim(top = 35, bottom = 0)
        plt.show(block=False)
        plt.pause(0.1)
    for i in reversed(range(24)):
        plt.cla()
        plt.plot(mat[:,i,0,1])
        plt.ylim(top = 35, bottom = 0)
        plt.show(block=False)
