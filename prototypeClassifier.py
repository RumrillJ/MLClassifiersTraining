import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import time

# step 1 create prototypes
prototype_A = np.array([
    [0, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1]
])

prototype_B = np.array([
    [1, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0]
])

prototype_C = np.array([
    [0, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 0]
])

prototype_D = np.array([
    [1, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0]
])

prototype_E = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1]
])

prototype_F = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0]
])

prototype_G = np.array([
    [0, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 0]
])

prototype_H = np.array([
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1]
])

prototype_I = np.array([
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0]
])

prototype_J = np.array([
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0]
])


def generate_noisy_variations(prototype, noise_percent, n_samples):
    variations = []
    numBits = prototype.size

    for _ in range(n_samples):
        noiseBits = np.copy(prototype.flatten())
        numNoiseBits = int(noise_percent * numBits)
        noiseIndex = np.random.choice(numBits, numNoiseBits, replace=False)
        noiseBits[noiseIndex] = 1 - noiseBits[noiseIndex]
        variations.append(noiseBits)

    return np.array(variations)


# noise_percent and n_samples
noise_percent = 0.3
n_samples = 30

prototypes = [prototype_A, prototype_B, prototype_C, prototype_D, prototype_E,
              prototype_F, prototype_G, prototype_H, prototype_I, prototype_J]

noisy_variations = []

for prototype in prototypes:
    variations = generate_noisy_variations(prototype, noise_percent, n_samples)
    noisy_variations.append(variations)

allVariations = np.concatenate(noisy_variations)
labels = np.repeat(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], n_samples)

# split prototype with 40% testing
X_train, X_test, y_train, y_test = train_test_split(allVariations, labels, test_size=0.4, random_state=42)

# classifiers MLP, k-NN, SVC, Random Forest from sklearn library
classifiers = {
    'MLP': MLPClassifier(max_iter=2000),
    'k-NN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}

best_classifier = ''
best_accuracy = 0.0

# look through classifiers to get accuracy and training time of each
for clf_name, clf in classifiers.items():
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{clf_name} Accuracy: {accuracy}")
    print(f"{clf_name} Training Time: {training_time} seconds")

    if accuracy > best_accuracy:
        best_classifier = clf
        best_accuracy = accuracy

# get best classifiers
print("Best Classifier:", best_classifier)


# select 3 best inputs using selectKBest
feature_selector = SelectKBest(score_func=chi2, k=3)
feature_selector.fit(X_train, y_train)
X_train_selected = feature_selector.transform(X_train)
X_test_selected = feature_selector.transform(X_test)

print("Selected Features:", feature_selector.get_support(indices=True))