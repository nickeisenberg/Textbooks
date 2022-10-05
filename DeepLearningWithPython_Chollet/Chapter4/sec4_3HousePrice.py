# Listing 4.23 Loading the Boston housing dataset 
from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = (boston_housing.load_data())

# Listing 4.24 Normailizing the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# Listing 4.25 Model definition
from tensorflow import keras
from tensorflow.keras import layers
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Listing 4.26 K-fold validation 
import numpy as np
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores =[]
for i in range(k):
    print(f'Processing fold #{i}')
    val_data = train_data[i*num_val_samples : (i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples : (i+1)*num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[: i*num_val_samples],
         train_data[(i+1)*num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[: i*num_val_samples],
         train_targets[(i+1)*num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data,
              partial_train_targets,
              epochs=num_epochs,
              batch_size=16,
              verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

# Listing 4.27 Saving the validation logs at each fold 
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f'Processing fold #{i}')
    val_data = train_data[i*num_val_samples : (i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples : (i+1)*num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[: i*num_val_samples],
         train_data[(i+1)*num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[: i*num_val_samples],
         train_targets[(i+1)*num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data,
              partial_train_targets,
              epochs=num_epochs,
              batch_size=16,
              validation_data=(val_data, val_targets),
              verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

# Listing 4.28 Building the history of successive mean K-fold validation scores
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# Listing 4.29 Plotting validation scores
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Listing 4.30 PLotting validation scored, excluding the first 10 data points
truncated_mae_history = average_mae_history[10:]
plt.clf()
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


