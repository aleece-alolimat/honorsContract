import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit
from param import Param
import data_loading
import cnn


# Initialize parameters and load data
param = Param()
X, Y = data_loading.read_data(param)

# CNN expects input with shape: (samples, channels, time, 1)
X = np.expand_dims(X, axis=-1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=param.test_part, random_state=0, shuffle=True
)

# Validation split configuration
val = round(param.validation_part * x_train.shape[0])
shuffle_split = ShuffleSplit(n_splits=param.cross_val_iter, test_size=val, random_state=0)

val_results = []
test_results = []

# Monte Carlo cross-validation
for i, (train_idx, val_idx) in enumerate(shuffle_split.split(x_train), 1):
    print(f"{i} / {param.cross_val_iter}  cross-validation iteration")

    # Initialize and train the CNN
    model = cnn.CNN(x_train.shape[1], x_train.shape[2], param)
    validation_metrics = model.fit(x_train[train_idx], y_train[train_idx],
                                   x_train[val_idx], y_train[val_idx])
    val_results.append(validation_metrics)

    # Evaluate on the held-out test set
    test_metrics = model.evaluate(x_test, y_test)
    test_results.append(test_metrics)

# Report average performance across folds
print("\nClassifier: CNN")

avg_val_results = np.round(np.mean(val_results, axis=0) * 100, 2)
avg_val_results_std = np.round(np.std(val_results, axis=0) * 100, 2)

print("Averaged validation results with averaged std in brackets:")
print(f"AUC: {avg_val_results[0]} ({avg_val_results_std[0]})")
print(f"accuracy: {avg_val_results[1]} ({avg_val_results_std[1]})")
print(f"precision: {avg_val_results[2]} ({avg_val_results_std[2]})")
print(f"recall: {avg_val_results[3]} ({avg_val_results_std[3]})")

print("\n##############################\n")

avg_test_results = np.round(np.mean(test_results, axis=0) * 100, 2)
avg_test_results_std = np.round(np.std(test_results, axis=0) * 100, 2)

print("Averaged test results with averaged std in brackets: ")
print(f"AUC: {avg_test_results[0]} ({avg_test_results_std[0]})")
print(f"accuracy: {avg_test_results[1]} ({avg_test_results_std[1]})")
print(f"precision: {avg_test_results[2]} ({avg_test_results_std[2]})")
print(f"recall: {avg_test_results[3]} ({avg_test_results_std[3]})")
