import numpy as np
import math
import matplotlib.pyplot as plt


def GenerateRandomVectors(meanVector, covarianceMatrix, N, savedFileName):
    A = np.zeros((2, 2))
    A[0, 0] = math.sqrt(covarianceMatrix[0, 0])
    A[1, 0] = covarianceMatrix[1, 0] / math.sqrt(covarianceMatrix[0, 0])
    A[1, 1] = math.sqrt(covarianceMatrix[1, 1] - covarianceMatrix[0, 1] * covarianceMatrix[0, 1] /
                        covarianceMatrix[0, 0])
    print(A)

    vectorsNorm01 = np.random.randn(2, N)

    x = np.matmul(A, vectorsNorm01) + np.repeat(meanVector, N, 1)
    np.save(savedFileName, x)


def Score_M(file):
    z1 = np.load(file)
    # z1.mean()
    # return z1
    x = np.array([[np.sum(z1[0]) / 200], [np.sum(z1[1]) / 200]])
    # print(x)
    return x


def Score_B(file):
    x = np.load(file)
    X_calc = np.array([[np.mean(x[0])], [np.mean(x[1])]])

    X_val = np.dot(X_calc, np.transpose(X_calc))
    sum = 0
    for i in range(200):
        xi = np.array([[x[0][i]], [x[1][i]]])
        xxT = np.dot(xi, np.transpose(xi))
        xxT_MM = xxT - X_val
        sum += xxT_MM
    # print(sum / 200)
    # print(np.cov(x))
    return sum / 200



def Task1(first, second):
    N = 200
    M1 = np.array([[0], [1]])
    M2 = np.array([[-1], [-1]])
    B = np.array([[5, 2], [2, 2]])

    GenerateRandomVectors(M1, B, N, first)
    GenerateRandomVectors(M2, B, N, second)

    z1 = np.load(first)
    z2 = np.load(second)
    plt.plot(z1[0, :], z1[1, :], linestyle='none', marker='.')
    plt.plot(z2[0, :], z2[1, :], color='green', linestyle='none', marker='*')
    plt.show()


def Task2(first, second, third):
    N = 200
    M1 = np.array([[0], [1]])
    M2 = np.array([[-1], [-1]])
    M3 = np.array([[2], [1]])
    B1 = np.array([[5, 2], [2, 1]])
    B2 = np.array([[3, 2], [1, 4]])
    B3 = np.array([[4, 1], [2, 1]])

    GenerateRandomVectors(M1, B1, N, first)
    GenerateRandomVectors(M2, B2, N, second)
    GenerateRandomVectors(M3, B3, N, third)

    z1 = np.load(first)
    z2 = np.load(second)
    z3 = np.load(third)
    plt.plot(z1[0, :], z1[1, :], linestyle='none', marker='.')
    plt.plot(z2[0, :], z2[1, :], color='green', linestyle='none', marker='*')
    plt.plot(z3[0, :], z3[1, :], color='red', linestyle='none', marker='D')
    plt.show()


def Task3(file1, file2, file3, file4, file5):
    x1 = np.load(file1)
    x2 = np.load(file2)
    x3 = np.load(file3)
    x4 = np.load(file4)
    x5 = np.load(file5)

    print("#####First#####")
    print("[", np.mean(x1[0]), np.mean(x1[1]), "]", np.cov(x1))
    print("#####")
    print(Score_M(file1), Score_B(file1))

    print("#####Second#####")
    print("[", np.mean(x2[0]), np.mean(x2[1]), "]", np.cov(x2))
    print("#####")
    print(Score_M(file2), Score_B(file2))

    print("#####Third#####")
    print("[", np.mean(x3[0]), np.mean(x3[1]), "]", np.cov(x3))
    print("#####")
    print(Score_M(file3), Score_B(file3))

    print("#####Fourth#####")
    print("[", np.mean(x4[0]), np.mean(x4[1]), "]", np.cov(x4))
    print("#####")
    print(Score_M(file4), Score_B(file4))

    print("#####Fifth#####")
    print("[", np.mean(x5[0]), np.mean(x5[1]), "]", np.cov(x5))
    print("#####")
    print(Score_M(file5), Score_B(file5))


if __name__ == '__main__':
    file_2_1 = "arrayX2_1.npy"
    file_2_2 = "arrayX2_2.npy"

    file_3_1 = "arrayX3_1.npy"
    file_3_2 = "arrayX3_2.npy"
    file_3_3 = "arrayX3_3.npy"

    Task1(file_2_1, file_2_2)
    Task2(file_3_1, file_3_2, file_3_3)
    Task3(file_2_1, file_2_2, file_3_1, file_3_2, file_3_3)
