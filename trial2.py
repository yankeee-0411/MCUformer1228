import numpy as np
import re
from scipy.spatial import KDTree


def parse_chains(chains_text):
    """
    Parses the Evolution Chains from a multiline string.

    Args:
        chains_text (str): Multiline string containing chains.

    Returns:
        list of list of tuples: Each sublist represents a chain of points as (x, y, z).
    """
    chains = []
    # Split the text into individual lines
    lines = [line.strip()
             for line in chains_text.strip().split('\n') if line.strip()]
    for line in lines:
        # Use regex to find all [x y z] patterns
        points = re.findall(r"\[([\d.]+)\s+([\d.]+)\s+([\d.]+)\]", line)
        # Convert each point to a tuple of floats
        chain = [tuple(map(float, point)) for point in points]
        chains.append(chain)
    return chains


def build_kdtree(X_coords):
    """
    Builds a KDTree for efficient nearest-neighbor searches.

    Args:
        X_coords (numpy.ndarray): Array of points with shape (n_points, 3).

    Returns:
        KDTree: A KDTree built from X_coords.
    """
    return KDTree(X_coords)


def map_points_to_y(chains, X_coords, y, tree, tolerance=1e-3):
    """
    Maps each point in the chains to its corresponding y value from X.

    Args:
        chains (list of list of tuples): Evolution Chains.
        X_coords (numpy.ndarray): Array of points from X with shape (n_points, 3).
        y (numpy.ndarray): Array of y values corresponding to X.
        tree (KDTree): KDTree built from X_coords.
        tolerance (float): Maximum distance to consider a match.

    Returns:
        list of list of tuples: Each point in the chains is annotated with its y value.
    """
    annotated_chains = []
    for chain_idx, chain in enumerate(chains):
        annotated_chain = []
        for point in chain:
            # Query the KDTree for the nearest neighbor
            distance, index = tree.query(point, distance_upper_bound=tolerance)
            if distance != np.inf:
                y_val = y[index]
            else:
                y_val = 0.0  # Assign 0 if no match is found
            annotated_chain.append((point, y_val))
        annotated_chains.append(annotated_chain)
    return annotated_chains


def format_and_print_chains(annotated_chains):
    """
    Formats and prints the annotated Evolution Chains.

    Args:
        annotated_chains (list of list of tuples): Chains with y values.
    """
    print("Evolution Chains with y values:\n")
    for idx, chain in enumerate(annotated_chains):
        formatted_points = []
        for point, y_val in chain:
            formatted_point = f"[{point[0]}  {point[1]}  {point[2]} ]({y_val:.6f})"
            formatted_points.append(formatted_point)
        chain_str = " --> ".join(formatted_points)
        print(f"Chain {idx}: {chain_str}\n")


def main():
    # Define X and y as provided
    X = np.array([
        [26.0, 0.7, 0.6, 1.0],
        [22.0, 0.9, 0.8, 1.0],
        [22.0, 0.8, 0.8, 1.0],
        [22.0, 0.85, 0.6, 1.0],
        [24.0, 0.65, 0.7, 1.0],
        [22.0, 0.7, 0.65, 1.0],
        [26.0, 0.6, 0.85, 1.0],
        [24.0, 0.6, 0.85, 1.0],
        [26.0, 0.65, 0.8, 1.0],
        [26.0, 0.6, 0.6, 1.0],
        [22.0, 0.6, 0.9, 1.0],
        [20.0, 0.84, 0.86, 1.0],
        [20.0, 0.64, 0.71, 1.0],
        [22.0, 0.6, 0.76, 1.0],
        [24.0, 0.6, 0.9, 1.0],
        [24.0, 0.6, 0.86, 1.0],
        [22.0, 0.74, 0.86, 1.0],
        [20.0, 0.79, 0.66, 1.0],
        [24.0, 0.66, 0.9, 1.0],
        [20.0, 0.6, 0.65, 1.0],
        [22.0, 0.66, 0.9, 1.0],
        [20.0, 0.73, 0.72, 1.0],
        [22.0, 0.6, 0.7, 1.0],
        [22.0, 0.72, 0.84, 1.0],
        [24.0, 0.72, 0.84, 1.0],
        [22.0, 0.66, 0.76, 1.0],
        [20.0, 0.66, 0.71, 1.0],
        [20.0, 0.71, 0.7, 1.0],
        [20.0, 0.78, 0.78, 1.0],
        [22.0, 0.78, 0.78, 1.0],
        [22.0, 0.72, 0.77, 1.0],
        [22.0, 0.72, 0.7, 1.0],
        [20.0, 0.71, 0.76, 1.0],
        [24.0, 0.78, 0.76, 1.0],
        [20.0, 0.72, 0.84, 1.0],
        [22.0, 0.77, 0.74, 1.0],
        [22.0, 0.78, 0.77, 1.0],
        [26.0, 0.72, 0.7, 1.0],
        [22.0, 0.75, 0.68, 1.0],
        [22.0, 0.8, 0.71, 1.0],
        [24.0, 0.81, 0.74, 1.0],
        [24.0, 0.74, 0.65, 1.0],
        [24.0, 0.66, 0.76, 1.0],
        [24.0, 0.87, 0.68, 1.0],
        [22.0, 0.68, 0.71, 1.0],
        [24.0, 0.74, 0.77, 1.0],
    ])

    y = np.array([
        0.50999571, 0.58812637, 0.56574341, 0.5387848, 0.57768581, 0.56617154,
        0.59417877, 0.57101882, 0.54695412, 0.51083569, 0.56720902, 0.60805325,
        0.5862941, 0.53886841, 0.55206803, 0.56251918, 0.60085879, 0.56248296,
        0.55221049, 0.54537709, 0.55414837, 0.57583613, 0.57104966, 0.58961701,
        0.5844894, 0.5722931, 0.54961048, 0.55711724, 0.59305895, 0.56351911,
        0.58002099, 0.60167687, 0.57159918, 0.58300992, 0.59278781, 0.57320682,
        0.58352881, 0.55912725, 0.58634839, 0.58872432, 0.58307239, 0.52476604,
        0.54012793, 0.55780312, 0.56844315, 0.59452197
    ])

    # Define Evolution Chains as a multiline string
    chains_text = """
        Chain 0: [26.0  0.65  0.8] --> [24.0  0.6  0.86] --> [24.0  0.66  0.9] --> [24.0  0.72  0.84] --> [22.0  0.78  0.78] --> [20.0  0.72  0.84] --> [20.0  0.78  0.9]
        Chain 1: [24.0  0.65  0.7] --> [22.0  0.6  0.76] --> [22.0  0.6  0.7] --> [22.0  0.66  0.76] --> [22.0  0.72  0.7] --> [24.0  0.78  0.76] --> [26.0  0.72  0.7] --> [24.0  0.66  0.76]
        Chain 2: [26.0  0.7  0.6] --> [24.0  0.64  0.66]
        Chain 3: [26.0  0.6  0.85] --> [24.0  0.6  0.9] --> [24.0  0.66  0.9] --> [24.0  0.72  0.84] --> [22.0  0.78  0.78] --> [20.0  0.72  0.84] --> [20.0  0.78  0.9]
        Chain 4: [26.0  0.6  0.6] --> [24.0  0.6  0.66]
        Chain 5: [22.0  0.9  0.8] --> [20.0  0.84  0.86] --> [20.0  0.9  0.9]
        Chain 6: [22.0  0.7  0.65] --> [20.0  0.64  0.71] --> [20.0  0.6  0.65] --> [20.0  0.66  0.71] --> [22.0  0.72  0.77] --> [22.0  0.78  0.77] --> [22.0  0.8  0.71] --> [24.0  0.74  0.65] --> [22.0  0.68  0.71] --> [24.0  0.74  0.77] --> [26.0  0.68  0.71]
        Chain 7: [22.0  0.8  0.8] --> [22.0  0.74  0.86] --> [20.0  0.8  0.9]
        Chain 8: [22.0  0.85  0.6] --> [20.0  0.79  0.66] --> [20.0  0.73  0.72] --> [20.0  0.71  0.7] --> [20.0  0.71  0.76] --> [22.0  0.77  0.74] --> [22.0  0.75  0.68] --> [24.0  0.81  0.74] --> [24.0  0.87  0.68]
        Chain 9: [24.0  0.6  0.85] --> [22.0  0.6  0.9] --> [22.0  0.66  0.9] --> [22.0  0.72  0.84] --> [20.0  0.78  0.78] --> [20.0  0.72  0.84] --> [20.0  0.78  0.9]
    """

    # Step 1: Parse the Evolution Chains
    chains = parse_chains(chains_text)

    # Step 2: Build KDTree for X coordinates (first three columns)
    X_coords = X[:, :3]
    tree = build_kdtree(X_coords)

    # Step 3: Map each point in chains to its corresponding y value
    annotated_chains = map_points_to_y(
        chains, X_coords, y, tree, tolerance=1e-2)

    # Step 4: Format and Print the Chains with y values
    format_and_print_chains(annotated_chains)


if __name__ == "__main__":
    main()
