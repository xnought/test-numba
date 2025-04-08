import numpy as np
from time import time
from numba import jit, prange


@jit
def square_(x):
    for i in range(len(x)):
        x[i] *= x[i]


@jit
def matmul(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
    M, INNER, N = a.shape[0], a.shape[1], b.shape[1]
    for m in range(M):
        for n in range(N):
            for inner in range(INNER):
                out[m, n] += a[m, inner] * b[inner, n]


@jit()
def matmul2(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
    M, INNER, N = a.shape[0], a.shape[1], b.shape[1]
    for m in range(M):
        for inner in range(INNER):
            for n in range(N):
                out[m, n] += a[m, inner] * b[inner, n]


@jit(parallel=True)
def matmul3(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
    M, INNER, N = a.shape[0], a.shape[1], b.shape[1]
    for m in prange(M):
        for inner in range(INNER):
            _a = a[m, inner]
            for n in range(N):
                out[m, n] += _a * b[inner, n]


def timeit(func):
    t0 = time()
    func()
    t1 = time()
    return t1 - t0


def main():
    N = 10_000
    a = np.random.randn(N, N)

    print("@")
    print(timeit(lambda: a @ a))

    print("matmul")
    output = np.zeros_like(a)
    matmul(a, a, output)  # warmup
    output = np.zeros_like(a)
    print(timeit(lambda: matmul(a, a, output)))

    print("matmul2")
    output = np.zeros_like(a)
    matmul2(a, a, output)  # warmup
    output = np.zeros_like(a)
    print(timeit(lambda: matmul2(a, a, output)))

    print("matmul3")
    output = np.zeros_like(a)
    matmul3(a, a, output)  # warmup
    output = np.zeros_like(a)
    print(timeit(lambda: matmul3(a, a, output)))


if __name__ == "__main__":
    main()
