import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from Lab_4.Lab_4 import generate_random_vectors


M1 = np.array([[0], [0]])
M2 = np.array([[1], [1]])
M3 = np.array([[-1], [1]])
M4 = np.array([[0], [1]])
M5 = np.array([[0], [2]])

B1 = np.array([[0.03, 0], [0, 0.03]])
B2 = np.array([[0.02, 0.01], [0.01, 0.05]])
B3 = np.array([[0.02, -0.01], [-0.01, 0.05]])
B4 = np.array([[0.02, 0.0], [0.0, 0.02]])
B5 = np.array([[0.05, 0], [0, 0.03]])

N = 50


path_to_file_1 = "Files/arrayX2_1.npy"
path_to_file_2 = "Files/arrayX2_2.npy"
path_to_file_3 = "Files/arrayX3_1.npy"
path_to_file_4 = "Files/arrayX3_2.npy"
path_to_file_5 = "Files/arrayX3_3.npy"


def generate_data():
    generate_random_vectors(M1, B1, N, path_to_file_1)
    generate_random_vectors(M2, B2, N, path_to_file_2)
    generate_random_vectors(M3, B3, N, path_to_file_3)
    generate_random_vectors(M4, B4, N, path_to_file_4)
    generate_random_vectors(M5, B5, N, path_to_file_5)


def load_data():
    sample_1 = np.load(path_to_file_1)
    sample_2 = np.load(path_to_file_2)
    sample_3 = np.load(path_to_file_3)
    sample_4 = np.load(path_to_file_4)
    sample_5 = np.load(path_to_file_5)
    return [sample_1, sample_2, sample_3, sample_4, sample_5]


def get_typical_distance(M, n=0.5):
    sum_distance = 0
    for l in range(M.shape[0]):
        for j in range(l+1, M.shape[0]):
            sum_distance += np.linalg.norm(M[l] - M[j])
    return n * 2 * sum_distance / (M.shape[0] * (M.shape[0] - 1))


def max_min_distance_algorithm(x):
    samples = np.copy(x)

    x_mean = np.mean(samples, axis=1)
    d = np.array([np.linalg.norm(samples[:, i] - x_mean) for i in range(samples.shape[1])])
    index = np.argmax(d)
    m0 = samples[:, index]
    samples = np.delete(samples, index, axis=1)

    d = np.array([np.linalg.norm(samples[:, i] - m0) for i in range(samples.shape[1])])
    index = np.argmax(d)
    m1 = samples[:, index]

    M_array = np.array([m0, m1])
    print(M_array)

    samples = np.delete(samples, index, axis=1)

    classification_history = []
    d_max_min_array = []
    d_typical_array = []

    while True:
        classification = np.zeros((samples.shape[1], 2+samples.shape[0]))
        for i in range(samples.shape[1]):
            d = np.array([np.linalg.norm(samples[:, i] - M_array[j]) for j in range(M_array.shape[0])])
            l = np.argmin(d)
            # print(d)
            # print(l)
            classification[i, 0] = l
            classification[i, 1] = d[l]
            classification[i, 2:] = samples[:, i]

        index = np.argmax(classification[:, 1])

        ml = samples[:, index]
        dl = classification[index, 1]
        d_typical = get_typical_distance(M_array)
        d_max_min_array.append(dl)
        d_typical_array.append(d_typical)

        samples = np.delete(samples, index, axis=1)
        classification_history.append(classification)

        if dl > d_typical:
            M_array = np.append(M_array, [ml], axis=0)
        else:
            break

    colors = ["red", "green", "blue", "purple", "black"]

    for i, classification in enumerate(classification_history):
        color = [colors[int(l)] for l in classification[:, 0]]
        m = M_array[:i+2]

        plt.figure(figsize=(16, 9))
        plt.scatter(classification[:, 2], classification[:, 3], color=color)
        plt.scatter(m[:, 0], m[:, 1], color='black', marker="x", s=150)
        plt.show()

    plt.figure(figsize=(16, 9))
    plt.suptitle("Зависимость d_max_min и d_typical от числа кластеров")
    plt.plot(np.arange(len(d_max_min_array)), d_max_min_array, color='black')
    plt.plot(np.arange(len(d_typical_array)), d_typical_array, color='red')
    plt.show()


def print_k_intergroup_averages_clusters(samples, cluster_numbers):
    for i, sample in enumerate(samples):
        if cluster_numbers[i] == 0:
            plt.scatter(samples[i, 0], samples[i, 1], color='cyan')
        elif cluster_numbers[i] == 1:
            plt.scatter(samples[i, 0], samples[i, 1], color='orange')
        elif cluster_numbers[i] == 2:
            plt.scatter(samples[i, 0], samples[i, 1], color='gray')
        elif cluster_numbers[i] == 3:
            plt.scatter(samples[i, 0], samples[i, 1], color='red')
        elif cluster_numbers[i] == 4:
            plt.scatter(samples[i, 0], samples[i, 1], color='pink')
    # plt.show()


def algorithm_k_intergroup_averages(samples, n, k, mode="ideal"):
    assert k < n
    assert mode in ["ideal", "notideal"]

    r = 1
    current_centers = np.zeros((k, samples.shape[1]))

    if mode == "ideal":
        for i in range(k):
            current_centers[i, :] = samples[i]
    elif (mode == "notideal") and (k == 5):
        current_centers[0] = np.array([1, 1])
        current_centers[1] = np.array([0.5, 0.2])
        current_centers[2] = np.array([0.7, 0.7])
        current_centers[3] = np.array([0.2, 1])
        current_centers[4] = np.array([0, 0])

    print(current_centers)

    cluster_numbers = np.zeros(samples.shape[0])
    cluster_change = np.empty((0, 2))

    while True:
        r += 1
        previous_centers = current_centers
        previous_cluster_numbers = np.copy(cluster_numbers)

        for i, sample in enumerate(samples):

            d = np.array([np.linalg.norm(center - sample) for center in previous_centers])
            index = np.argmin(d)
            cluster_numbers[i] = index

        sizes = Counter(cluster_numbers)
        print(sizes)
        sums = np.zeros((k, samples.shape[1]))

        for i in range(k):
            for j, sample in enumerate(samples):
                if cluster_numbers[j] == i:
                    sums[i] += sample
            sums[i] /= sizes[i]

        current_centers = sums

        count = 0
        for i in range(samples.shape[0]):
            if previous_cluster_numbers[i] != cluster_numbers[i]:
                count += 1

        cluster_change = np.vstack([cluster_change, np.array([r, count])])

        d = np.array([np.linalg.norm(current_centers[i] - previous_centers[i]) for i in range(current_centers.shape[0])])
        if np.max(d) < 0.0001:
            break

    print(f"Смена кластера: \n {cluster_change}")
    plt.figure(figsize=(12, 12))
    plt.plot(cluster_change[:, 0], cluster_change[:, 1])
    plt.show()

    colors = ["red", "green", "blue", "purple", "black"]
    color = [colors[int(cluster_number)] for cluster_number in cluster_numbers]

    plt.figure(figsize=(16, 9))
    plt.scatter(samples[:, 0], samples[:, 1], color=color)

    for i, center in enumerate(current_centers):
        plt.scatter(current_centers[i, 0], current_centers[i, 1], color='black', marker="x")
        plt.text(float(current_centers[i, 0]), float(current_centers[i, 1]), i, fontsize=12)

    plt.show()


def task_1():
    samples = load_data()
    colors = ["red", "green", "blue", "purple", "black"]

    for i, sample in enumerate(samples):
        plt.scatter(sample[0, :], sample[1, :], color=colors[i])
    plt.show()


def task_2():
    samples = load_data()
    x = np.concatenate(samples, axis=1)
    return x


def main():
    generate_data()
    # task_1()
    x = task_2()
    # max_min_distance_algorithm(x)
    algorithm_k_intergroup_averages(x.T, 250, 5, mode="notideal")


if __name__ == "__main__":
    main()
