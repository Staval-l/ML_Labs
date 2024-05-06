import numpy as np
import math
import matplotlib.pyplot as plt


B0 = np.array([[0.35, 0.15], [0.15, 0.35]])  # Тут была неправильная корр. матрица


def GenerateRandomVectors(meanVector, covarianceMatrix, N, savedFileName):
    # A = np.zeros((2, 2))
    # A[0, 0] = math.sqrt(covarianceMatrix[0, 0])
    # A[1, 0] = covarianceMatrix[1, 0] / math.sqrt(covarianceMatrix[0, 0])
    # A[1, 1] = math.sqrt(covarianceMatrix[1, 1] - covarianceMatrix[0, 1] * covarianceMatrix[0, 1] /
    #                     covarianceMatrix[0, 0])
    # print(A)
    #
    # vectorsNorm01 = np.random.randn(2, N)
    # x = np.matmul(A, vectorsNorm01) + np.repeat(meanVector, N, 1)
    # np.save(savedFileName, x)

    vectorsNorm01 = np.array([np.zeros(N), np.zeros(N)])

    for i in range(50):
        uni = np.array([np.random.uniform(0, 6, N), np.random.uniform(0, 6, N)])
        vectorsNorm01 += uni
    vectorsNorm01 /= 50

    m = [[3], [3]]
    d = [[3], [3]]

    vectorsNorm01 = np.sqrt(50) * (vectorsNorm01 - np.repeat(m, N, 1)) / np.sqrt(d)

    # vectorsNorm01 = np.random.randn(2, N)

    A = np.zeros((2, 2))
    A[0, 0] = math.sqrt(covarianceMatrix[0, 0])
    A[1, 0] = covarianceMatrix[1, 0] / math.sqrt(covarianceMatrix[0, 0])
    A[1, 1] = math.sqrt(covarianceMatrix[1, 1] - covarianceMatrix[0, 1] * covarianceMatrix[0, 1]
                        / covarianceMatrix[0, 0])

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


def CalcM(z):
    return np.array([[np.mean(z[0])], [np.mean(z[1])]])


def Dist_B(first, second):  # Неправильная реализация!
    x1 = np.load(first)
    x2 = np.load(second)
    return ((0.25 * np.transpose(CalcM(x2) - CalcM(x1)) *
            np.power((np.cov(x1) + np.cov(x2)) / 2, -1)) * (CalcM(x2) - CalcM(x1)) +
            0.5 * np.log(np.abs((np.cov(x1) + np.cov(x2)) / 2)) / np.sqrt(np.cov(x1) * np.cov(x2)))


# Только если корр. матрицы равны!
def Dist_M(first, second, B):  # Неправильная реализация!
    x1 = np.load(first)
    x2 = np.load(second)
    return np.transpose(CalcM(x2) - CalcM(x1)) * B * (CalcM(x2) - CalcM(x1))


# Правильная реализация:
# def Mahalanobis_distance(M0, M1, B):
#     return (M1 - M0) @ np.linalg.inv(B) @ np.transpose(M1 - M0)


def Task1(first, second):
    N = 200
    M1 = np.array([[0], [1]])
    M2 = np.array([[-1], [-1]])
    B = np.array([[0.35, 0.15], [0.1, 0.35]])

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
    B1 = np.array([[0.45, 0.15], [0.15, 0.45]])
    B2 = np.array([[0.15, 0.02], [0.02, 0.15]])
    B3 = np.array([[0.25, -0.17], [-0.17, 0.25]])

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
    file_2_1 = "Files/arrayX2_1.npy"
    file_2_2 = "Files/arrayX2_2.npy"

    file_3_1 = "Files/arrayX3_1.npy"
    file_3_2 = "Files/arrayX3_2.npy"
    file_3_3 = "Files/arrayX3_3.npy"

    Task1(file_2_1, file_2_2)
    Task2(file_3_1, file_3_2, file_3_3)
    Task3(file_2_1, file_2_2, file_3_1, file_3_2, file_3_3)

    print(Dist_B(file_2_1, file_2_2))
    print(Dist_M(file_2_1, file_2_2, B0))
