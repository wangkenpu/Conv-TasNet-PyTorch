# A Quick Implementation of UPGMA (Unweighted Pair Group Method with Arithmetic Mean)

import re
import numpy as np


def euclidean_metric(x, y):
    return np.linalg.norm(x - y)


def matrix_sort(matrix):
    rows, _ = matrix.shape
    new_matrix = np.copy(matrix)
    row_sum = np.sum(matrix, axis=1)
    cursor = np.argsort(row_sum)
    for i in range(rows):
        new_matrix[i] = matrix[cursor[i]]
    return new_matrix


def construct_table(matrix):
    row, col = matrix.shape
    table = []
    for i in range(row):
        dist = []
        for j in range(i):
           cost = euclidean_metric(matrix[i], matrix[j])
           dist.append(cost)
        table.append(dist)
    return table


# lowest_cell:
#   Locates the smallest cell in the table
def lowest_cell(table):
    # Set default to infinity
    min_cell = float("inf")
    x, y = -1, -1

    # Go through every cell, looking for the lowest
    for i in range(len(table)):
        for j in range(len(table[i])):
            if table[i][j] < min_cell:
                min_cell = table[i][j]
                x, y = i, j

    # Return the x, y co-ordinate of cell
    return x, y


# join_labels:
#   Combines two labels in a list of labels
def join_labels(labels, a, b):
    # Swap if the indices are not ordered
    if b < a:
        a, b = b, a

    # Join the labels in the first index
    labels[a] = "(" + str(labels[a]) + "," + str(labels[b]) + ")"

    # Remove the (now redundant) label in the second index
    del labels[b]


# join_table:
#   Joins the entries of a table on the cell (a, b) by averaging their data entries
def join_table(table, a, b):
    # Swap if the indices are not ordered
    if b < a:
        a, b = b, a

    # For the lower index, reconstruct the entire row (A, i), where i < A
    row = []
    for i in range(0, a):
        row.append((table[a][i] + table[b][i]) / 2)
    table[a] = row

    # Then, reconstruct the entire column (i, A), where i > A
    # NOTE: Since the matrix is lower triangular, row b only contains values for indices < b
    for i in range(a + 1, b):
        table[i][a] = (table[i][a] + table[b][i]) / 2

    # We get the rest of the values from row i
    for i in range(b + 1, len(table)):
        table[i][a] = (table[i][a] + table[i][b]) / 2
        # Remove the (now redundant) second index column entry
        del table[i][b]

    # Remove the (now redundant) second index row
    del table[b]


# UPGMA:
#   Runs the UPGMA algorithm on a labelled table
def UPGMA(table, labels):
    # Until all labels have been joined...
    while len(labels) > 1:
        # Locate lowest cell in the table
        x, y = lowest_cell(table)

        # Join the table on the cell co-ordinates
        join_table(table, x, y)

        # Update the labels accordingly
        join_labels(labels, x, y)

    # Return the final label
    return labels[0]



## A test using an example calculation from http://www.nmsr.org/upgma.htm

# alpha_labels:
#   Makes labels from a starting letter to an ending letter
def alpha_labels(start, end):
    labels = []
    for i in range(ord(start), ord(end)+1):
        labels.append(chr(i))
    return labels


def numeric_lables(start, end):
    labels = []
    for i in range(start, end):
        labels.append(i)
    return labels


# Test table data and corresponding labels
M_labels = alpha_labels("A", "G")   #A through G
M = [
    [],                         #A
    [19],                       #B
    [27, 31],                   #C
    [8, 18, 26],                #D
    [33, 36, 41, 31],           #E
    [18, 1, 32, 17, 35],        #F
    [13, 13, 29, 14, 28, 12]    #G
    ]

def perform_UPGMA(matrix):
    new_matrix = np.copy(matrix)
    matrix = matrix_sort(matrix)
    dist = construct_table(matrix)
    labels = numeric_lables(0, len(dist))
    order_list = get_order(UPGMA(dist, labels))
    for idx, item in enumerate(order_list):
        new_matrix[idx] = matrix[int(item)]
    return new_matrix


def get_order(tree):
    order_list = re.split(',', tree)
    for idx, item in enumerate(order_list):
        item = item.strip('(').strip(')')
        order_list[idx] = item
    return order_list


# NUM = 10
# matrix = np.ones((NUM, NUM))
# row, col = matrix.shape
# rand_int = [i + 1 for i in range(row)]
# import random
# random.shuffle(rand_int)
# for i in range(row):
#     matrix[i] = matrix[i] * rand_int[i]
# print(matrix)
# matrix = matrix_sort(matrix)
# print(matrix)
# dist = construct_table(matrix)
# labels = numeric_lables(0, NUM)
# print(UPGMA(dist, labels))

# UPGMA(M, M_labels) should output: '((((A,D),((B,F),G)),C),E)'
# print(UPGMA(M, M_labels))
