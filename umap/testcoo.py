""" Some playing around with SciPy's coo matrices"""
import codeop

import numpy as np
from scipy.sparse import coo_matrix


row = np.array([0, 0, 1, 3, 1, 0, 0, 2])
col = np.array([0, 2, 1, 3, 1, 0, 0, 1])
data = np.array([1, 1, 1, 1, 1, 1, 1, 10])

coo = coo_matrix((data, (row, col)), shape=(4, 5))

# Let's reproduce step by step what `sum_duplicates` does

print("coo matrix before calling .sum_duplicates()\n", coo, "\n---")

coo.sum_duplicates()

print("coo matrix after calling .sum_duplicates()\n", coo)


print((row, col))

# lexsort sorts starting by the last array. In a dictionary, "col" would have the first letter of a word,
# and row would have the second letter of the word
# It returns the indices that contain the order

order = np.lexsort((row, col))

# Reorder the elements
row = row[order]
col = col[order]
data = data[order]

print(row)
print(row[1:])
print(row[:-1])

print(row[1:] != row[:-1])

print(col)
print(col[1:])
print(col[:-1])
print(col[1:] != col[:-1])

print(((row[1:] != row[:-1]) | (col[1:] != col[:-1])))
unique_mask = ((row[1:] != row[:-1]) | (col[1:] != col[:-1]))
print(unique_mask)
print(np.append(True, unique_mask))

print(len(coo.shape))