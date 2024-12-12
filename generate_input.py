import random
import sys


def generate_points(n_points, dims, output_file_name):
    with open(output_file_name, 'w') as f:
        f.write("{}\n".format(n_points))
        for i in range(1, n_points + 1):
            features = [random.uniform(0, 1) for _ in range(dims)]
            f.write(f"{i} {' '.join(map(str, features))}\n")


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python generate_inputs.py <n_points> <dims>")
    #     sys.exit(1)

    # n_points = int(sys.argv[1])
    # dims = int(sys.argv[2])

    n_points_list = [2**i for i in range(10,21)]
    dims = 32
    for n_points in n_points_list:
        output_file_name = "input/input-n{}-d{}-c16.txt".format(n_points, dims)
        generate_points(n_points, dims, output_file_name)