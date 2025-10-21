import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import time

def load_data(file_path, x_key='x_train', y_key='y_train'):
    """Load input and output data from a .mat file."""
    data = scipy.io.loadmat(file_path)
    return data[x_key], data[y_key]

def build_model(input_dim, hidden_layers=5, units=64, activation='tanh'):
    """Create a fully connected Sequential model."""
    model = Sequential()
    # Add hidden layers
    for i in range(hidden_layers):
        if i == 0:
            model.add(Dense(units=units, input_shape=(input_dim,), activation=activation))
        else:
            model.add(Dense(units=units, activation=activation))
    # Add output layer
    model.add(Dense(units=1, activation='linear'))
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=1000, batch_size=256, patience=1000):
    """Train the model with early stopping."""
    early_stopping = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        shuffle=True, callbacks=[early_stopping])
    elapsed = time.time() - start_time
    return history, elapsed

def predict_and_plot(model, X_test, y_test, save_path='predictions.csv'):
    """Make predictions, save to CSV, and plot actual vs predicted values."""
    predictions = model.predict(X_test)
    pd.DataFrame(predictions, columns=['Prediction']).to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual', marker='o')
    plt.plot(predictions, label='Predicted', marker='x')
    plt.title('Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# ===== Main Script =====
train_file = 'D5.mat'

# Load training data
X_train, y_train = load_data(train_file, x_key='x_train', y_key='y_train')

# Build model
model = build_model(input_dim=X_train.shape[1])

# Print model summary
model.summary()

# Train model
history, elapsed_time = train_model(model, X_train, y_train)
print(f"Training time: {elapsed_time:.2f} seconds")

# Save model
model.save('D5.h5')

# Load test data
X_test, y_test = load_data(train_file, x_key='x_test', y_key='y_test')

# Predict and plot
predict_and_plot(model, X_test, y_test)
