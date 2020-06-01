import numpy as np


def get_features_and_labels(model, generator, count):
    features = []
    labels = []
    for i in range(count):
        try:
            x, y = next(generator)
        except StopIteration as _:
            break

        if len(y.shape) == 2:
            y = np.argmax(y, axis=-1)

        features.append(model(x))
        labels.append(y)

    return np.vstack(features), np.hstack(labels)
