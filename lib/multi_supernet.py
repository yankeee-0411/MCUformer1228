import itertools
import numpy as np
import random
from pathlib import Path
from scipy.spatial import KDTree

def make_list(input1, input2):
    return list(itertools.product(input1, input2))

def find_max_positions(lst):
    max_val = max(lst)
    max_positions = []
    for i, val in enumerate(lst):
        if val == max_val:
            max_positions.append(i)
    return max_positions[-1]

def find_min_positions(lst):
    min_val = min(lst)
    min_positions = []
    for i, val in enumerate(lst):
        if val == min_val:
            min_positions.append(i)
    return min_positions[0]

# def create_list(point_list):
#     x = random.sample(point_list, 3)
#     l = len(point_list)
#     pos_choice = []
#     for i in range(l):
#         pos_choice.append(point_list[i][0] + point_list[i][1])
#     x_max_pos = find_max_positions(pos_choice)
#     x_min_pos = find_min_positions(pos_choice)
#     x.append(point_list[x_max_pos])
#     x.append(point_list[x_min_pos])
#     return x
# 
# def create_list(point_list):
#     x = random.sample(point_list, 5)
#     print(x)
#     return x

def find_nearlist_point(supernet, current_point, distance):
    tree = KDTree(supernet)
    _, indices = tree.query(current_point, distance)
    choose_point = []
    for item in indices:
        choose_point.append(supernet[item])
    return choose_point

# data = [[1, 2, 10], [3, 4, 10], [5, 6, 10], [6, 8, 10], [9, 10, 10]]
# print(find_nearlist_point(data, [4,5], 5))

class evolution_supernet(object):
    def __init__(self, result_array, rank_ratio, patch_size):
        self.result_array = result_array
        self.rank_ratio = rank_ratio
        self.patch_size = patch_size

    def fit_SRAM_plane(self):
        # Create a 2D array X, where each row is [1, x1, x2] from the result_array
        X = np.array([[1, x1, x2] for x1, x2, y1, y2 in self.result_array])
        
        # Create a 1D array y, where each element is 1-y2 from the result_array
        y = np.array([[1-y2] for x1, x2, y1, y2 in self.result_array]).reshape(-1, 1)
        
        # Calculate the weights w by solving the normal equation: w = (X^T * X)^-1 * X^T * y
        w = np.linalg.inv(X.T @ X) @ X.T @ y
        
        # Return the last two elements of the reshaped weight vector
        return w.reshape(-1)[-2:]

    def get_SRAM_evolution_step(self):
        weight_array = self.fit_SRAM_plane()
        print("SRAM", weight_array)
        threshold_array = np.array([1.5, 0.3])
        step_array = np.array([0.05, 4])
        return np.around(weight_array/threshold_array) *step_array


    def fit_error_plane(self):
        X = np.array([[1, x1, x2] for x1, x2, y1, y2 in self.result_array])
        y = np.array([[y1] for x1, x2, y1, y2 in self.result_array]).reshape(-1, 1)

        w = np.linalg.inv(X.T @ X) @ X.T @ y
        return w.reshape(-1)[-2:]

    def get_error_evolution_step(self):
        weight_array = self.fit_error_plane()
        print("error", weight_array)
        threshold_array = np.array([0.25, 0.01])
        step_array = np.array([0.05, 4])
        return np.around(weight_array/threshold_array) *step_array
    
    def evolution_step(self):
        error_step = self.get_error_evolution_step()
        sram_step = self.get_SRAM_evolution_step()
        step =  (error_step - sram_step)
        print(step)
        if self.patch_size + step[1] < 16:
            step[1] = 16 - self.patch_size
        if self.patch_size + step[1] > 32:
            step[1] = 32 - self.patch_size
        if self.rank_ratio + step[0] < 0.4:
            step[0] = 0.4 - self.rank_ratio
        if self.rank_ratio + step[0] > 0.95:
            step[0] = 0.95 - self.rank_ratio
        return step

[[0.9, 16, 0.664, 0.384], [0.6, 16, 0.6054, 0.66], [0.85, 20, 0.625, 1], [0.95, 28, 0.576, 1], [0.65, 28, 0.454, 1]]

[[0.95, 16, 0.654, 0.342], [0.9, 16, 0.664, 0.384], [0.85, 20, 0.625, 1], [0.95, 28, 0.576, 1], [0.95, 24, 0.604, 1]]
[[0.65, 16, 0.61, 0.689], [0.6, 16, 0.6054, 0.66], [0.85, 20, 0.625, 1], [0.65, 28, 0.454, 1], [0.7, 24, 0.534, 1]]
[[0.95, 16, 0.654, 0.342], [0.9, 16, 0.664, 0.384], [0.85, 20, 0.625, 1], [0.95, 28, 0.576, 1], [0.95, 24, 0.62, 1]]
[[0.65, 16, 0.61, 0.689], [0.6, 16, 0.6054, 0.66], [0.85, 20, 0.625, 1], [0.65, 28, 0.454, 1], [0.7, 24, 0.534, 1]]
[[0.95, 16, 0.654, 0.342], [0.9, 16, 0.664, 0.384], [0.85, 20, 0.625, 1], [0.95, 28, 0.576, 1], [0.95, 24, 0.604, 1]]

[[0.9, 16, 0.664, 0.342], [0.85, 16, 0.655, 0.384], [0.65, 16, 0.61, 0.689], [0.8, 20, 0.623, 1], [0.85, 20, 0.625, 1]]
[[0.9, 16, 0.664, 0.342], [0.85, 16, 0.655, 0.384], [0.85, 20, 0.625, 1]]
[[0.75, 16, 0.553, 0.342], [0.85, 16, 0.655, 0.384], [0.75, 16, 0.553, 0.342], [0.8, 20, 0.623, 1], [0.85, 20, 0.625, 1]]

[[0.95, 16, 0.654, 0.342], [0.9, 16, 0.664, 0.384], [0.95, 20, 0.632, 0.63], [0.9, 20, 0.622, 1], [0.95, 24, 0.612, 1]]
[[0.9, 16, 0.664, 0.384], [0.95, 20, 0.632, 0.63], [0.9, 20, 0.622, 1], [0.85, 20, 0.625, 1], [0.85, 16, 0.655, 0.384]]
[[0.95, 16, 0.654, 0.342], [0.9, 16, 0.664, 0.384], [0.85, 16, 0.655, 0.384], [0.9, 20, 0.622, 1],  [0.85, 20, 0.625, 1]]




[[0.85, 20, 0.625, 1], [0.55, 16, 0.572, 0.688], [0.6, 16, 0.6054, 0.66], [0.8, 16, 0.657, 0.342], [0.85, 16, 0.654, 0.342]]
[[0.8, 16, 0.657, 0.342], [0.9, 16, 0.664, 0.384], [0.85, 20, 0.625, 1], [0.8, 16, 0.657, 0.342], [0.9, 24, 0.614, 1]]
[[0.85, 16, 0.654, 0.342], [0.9, 16, 0.664, 0.384], [0.85, 20, 0.625, 1], [0.95, 28, 0.576, 1], [0.9, 24, 0.614, 1]]
[[0.55, 16, 0.572, 0.688], [0.6, 16, 0.6054, 0.66], [0.85, 20, 0.625, 1], [0.65, 28, 0.454, 1], [0.6, 24, 0.42, 1]]



# input_vectors =[[0.95, 16, 0.654, 0.342], [0.9, 16, 0.664, 0.384], [0.85, 16, 0.655, 0.384], [0.9, 20, 0.622, 1],  [0.85, 20, 0.625, 1]]
# print(evolution_supernet(input_vectors, 0.75, 24).evolution_step())

input_vectors = [[0.4, 16, 0.461, 0.66], [0.7, 16, 0.624, 0.30],[0.8, 16, 0.651, 0.32],[0.4, 24, 0.404, 1],[0.5, 24, 0.456, 1]]

print(evolution_supernet(input_vectors, 0.4, 16).evolution_step())
print(evolution_supernet(input_vectors, 0.7, 16).evolution_step())
print(evolution_supernet(input_vectors, 0.8, 16).evolution_step())
print(evolution_supernet(input_vectors, 0.4, 24).evolution_step())
print(evolution_supernet(input_vectors, 0.5, 24).evolution_step())
input_vectors = [[0.45, 16, 0.498, 0.66], [0.75, 16, 0.635, 0.32], [0.85, 16, 0.653, 0.32], [0.45, 20, 0.461, 1], [0.55, 20, 0.549, 1]]
print(evolution_supernet(input_vectors, 0.45, 16).evolution_step())
print(evolution_supernet(input_vectors, 0.75, 16).evolution_step())
print(evolution_supernet(input_vectors, 0.85, 16).evolution_step())
print(evolution_supernet(input_vectors, 0.45, 20).evolution_step())
print(evolution_supernet(input_vectors, 0.55, 20).evolution_step())