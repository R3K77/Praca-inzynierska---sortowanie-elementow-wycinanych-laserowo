from visualisation_nc_file import visualize_cutting_paths
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import re

def compare_shapes(file_path, x_max=500, y_max=1000):
    cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths(file_path)
    num_shapes = len(cutting_paths)
    similarity_results = []  # List to store the comparison results

    for i in range(num_shapes):
        shape_similarity = []  # List to store the comparison results for the current shape
        for j in range(num_shapes):  # Compare the current shape with all shapes, including itself
            if i == j:
                shape_similarity.append(True)  # A shape is always similar to itself
                continue

            shape_1 = np.array(cutting_paths[i], dtype=np.int32)  # Convert to numpy array and reshape
            shape_2 = np.array(cutting_paths[j], dtype=np.int32)  # Convert to numpy array and reshape

            # Compare shapes using cv2.matchShapes()
            match_score = cv2.matchShapes(shape_1, shape_2, cv2.CONTOURS_MATCH_I1, 0.0)

            # Decide whether the shapes are similar based on the match score
            is_similar = match_score < 0.1  # Change this threshold as needed
            shape_similarity.append((j, is_similar))

        similarity_results.append(shape_similarity)

    return similarity_results

if __name__ == "__main__":
    porownanie = compare_shapes("NC_files/Arkusz-1001.nc")
    print(porownanie[7])
