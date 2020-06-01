import numpy as np


class SphericalKMeans:
    def __init__(self, max_iter=300, tol=1e-4, eps=1e-8):
        self.__max_iter = max_iter
        self.__tol = tol
        self.__eps = eps

        self.__X = None
        self.__centers, self.__prev_centers = None, None
        self.__labels = None

    def fit_predict(self, x, init):
        x = np.asarray(x, dtype=np.float32)  # [objects, dim]
        init = np.asarray(init, dtype=np.float32)  # [centers, dim]

        assert x.ndim == 2 and init.ndim == 2
        assert x.shape[1] == init.shape[1]

        self.__X = x / (np.linalg.norm(x, axis=-1, keepdims=True) + self.__eps)
        self.__centers = init / (np.linalg.norm(init, axis=-1, keepdims=True) + self.__eps)
        self.__labels = np.empty(len(x), dtype=np.int32)

        convergence = False
        for _ in range(self.__max_iter):
            self.__update_labels()
            self.__update_centers()

            if self.__should_stop():
                convergence = True
                break

        return self.__labels, self.__centers, convergence

    def __update_labels(self):
        pairwise_distances = (1 - np.matmul(self.__X, self.__centers.T))  # [objects, centers]
        np.argmin(pairwise_distances, axis=1, out=self.__labels)

    def __update_centers(self):
        for class_ix in range(self.__centers.shape[0]):
            self.__centers[class_ix] = np.sum(self.__X[self.__labels == class_ix], axis=0)
        self.__centers = self.__centers / (np.linalg.norm(self.__centers, axis=1, keepdims=True) + self.__eps)

    def __should_stop(self):
        if self.__prev_centers is None:
            self.__prev_centers = self.__centers.copy()
            return False

        should_stop = np.allclose(self.__prev_centers, self.__centers, rtol=self.__tol)
        self.__prev_centers = self.__centers.copy()
        return should_stop
