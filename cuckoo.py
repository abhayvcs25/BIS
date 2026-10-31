import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

n_features = X.shape[1]

# 2. Fitness function (paper's Psi = 0.6)
def fitness(solution):
    selected = np.where(solution == 1)[0]

    if len(selected) == 0:
        return 999

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    accuracy = cross_val_score(
        model, X_train[:, selected], y_train,
        cv=cv, scoring="accuracy"
    ).mean()

    feature_ratio = len(selected) / n_features

    # Convert paper's maximization objective into minimization
    score = 0.6 * accuracy + 0.4 * (1 - feature_ratio)

    return 1 - score


# 3. BRCSA parameters
N = 10
MAX_ITER = 15
Pa = 0.25
Psi = 0.6

# Initialize binary nests
nests = np.random.randint(0, 2, (N, n_features))
fitness_values = np.array([fitness(x) for x in nests])

best_idx = np.argmin(fitness_values)
best_nest = nests[best_idx].copy()
best_fitness = fitness_values[best_idx]

# Binary Novel External Archive
archive = [best_nest.copy()]

# 4. Optimization
for iteration in range(MAX_ITER):

    # Generate new solutions using a binary search move
    for i in range(N):

        new_nest = nests[i].copy()

        # Random binary changes guided by the best solution
        for j in range(n_features):
            if np.random.rand() < 0.1:
                new_nest[j] = best_nest[j]

        new_fit = fitness(new_nest)

        # Greedy selection: preserve solution and fitness together
        if new_fit < fitness_values[i]:
            nests[i] = new_nest.copy()
            fitness_values[i] = new_fit

    # Discovery and abandonment
    mean_fit = np.mean(fitness_values)
    threshold = best_fitness + 0.2 * (mean_fit - best_fitness)

    for i in range(N):
        if fitness_values[i] > threshold and np.random.rand() < Pa:

            # Create a replacement using two randomly selected nests
            a, b = np.random.choice(N, 2, replace=False)
            new_nest = np.logical_xor(nests[a], nests[b]).astype(int)

            new_fit = fitness(new_nest)

            if new_fit < fitness_values[i]:
                nests[i] = new_nest.copy()
                fitness_values[i] = new_fit

    # Update best solution
    best_idx = np.argmin(fitness_values)

    if fitness_values[best_idx] < best_fitness:
        best_fitness = fitness_values[best_idx]
        best_nest = nests[best_idx].copy()

    # Store diverse solutions in the external archive
    for nest in nests:
        if all(np.mean(nest != archived) > 0.5 for archived in archive):
            archive.append(nest.copy())

    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness:.4f}")


# 5. Final evaluation
selected = np.where(best_nest == 1)[0]

if len(selected) == 0:
    raise RuntimeError("No features were selected.")

baseline = make_pipeline(
    StandardScaler(), KNeighborsClassifier(n_neighbors=5)
)
optimized = make_pipeline(
    StandardScaler(), KNeighborsClassifier(n_neighbors=5)
)

baseline.fit(X_train, y_train)
optimized.fit(X_train[:, selected], y_train)

acc_all = accuracy_score(y_test, baseline.predict(X_test))
acc_selected = accuracy_score(
    y_test, optimized.predict(X_test[:, selected])
)

print("\n========== BRCSA RESULTS ==========")
print(f"Accuracy using all features: {acc_all * 100:.2f}%")
print(f"Accuracy using selected features: {acc_selected * 100:.2f}%")
print(f"Selected features: {len(selected)} / {n_features}")
print("Feature names:", data.feature_names[selected].tolist())
print("Archive solutions:", len(archive))