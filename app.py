"""
Advanced Neural Network Simulator
Diploma Project - Professional Edition
Supports: TensorFlow, PyTorch, Scikit-learn
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import json
import uuid
import os
from datetime import datetime
import io
import base64
from PIL import Image

# Deep Learning Frameworks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Machine Learning
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.datasets import make_classification, make_moons, make_circles

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = 'advanced_neural_network_simulator_2025_diploma'

# Store networks
networks = {}
training_sessions = {}

# ====================
# TENSORFLOW MODELS
# ====================

class TensorFlowNetwork:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'epoch': []}
        self.framework = 'tensorflow'
        self.build_model()
    
    def build_model(self):
        """Build TensorFlow/Keras model"""
        layers_config = self.config['layers']
        activation = self.config.get('activation', 'relu')
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(layers_config[0], activation=activation, 
                              input_shape=(layers_config[0],),
                              name='input_layer'))
        
        # Hidden layers
        for i, units in enumerate(layers_config[1:-1], 1):
            model.add(layers.Dense(units, activation=activation, 
                                  name=f'hidden_{i}'))
            
            # Add dropout if specified
            if self.config.get('dropout', 0) > 0:
                model.add(layers.Dropout(self.config['dropout']))
            
            # Add batch normalization if specified
            if self.config.get('batch_norm', False):
                model.add(layers.BatchNormalization())
        
        # Output layer
        output_activation = 'sigmoid' if layers_config[-1] == 1 else 'softmax'
        model.add(layers.Dense(layers_config[-1], activation=output_activation,
                              name='output_layer'))
        
        # Compile model
        optimizer_name = self.config.get('optimizer', 'adam')
        learning_rate = self.config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy' if layers_config[-1] == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
    
    def train(self, X, y, epochs=10, validation_split=0.2):
        """Train the model"""
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            verbose=0,
            batch_size=self.config.get('batch_size', 32)
        )
        
        # Update history
        for key in history.history.keys():
            if key in self.history:
                self.history[key].extend(history.history[key])
        
        self.history['epoch'].extend(range(len(self.history['epoch']) + 1, 
                                          len(self.history['epoch']) + epochs + 1))
        
        return {
            'loss': float(history.history['loss'][-1]),
            'accuracy': float(history.history['accuracy'][-1]),
            'val_loss': float(history.history.get('val_loss', [0])[-1]),
            'val_accuracy': float(history.history.get('val_accuracy', [0])[-1])
        }
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X, dtype=np.float32)
        predictions = self.model.predict(X, verbose=0)
        return predictions.tolist()
    
    def get_weights(self):
        """Get model weights"""
        weights = []
        for layer in self.model.layers:
            if isinstance(layer, layers.Dense):
                w = layer.get_weights()[0]
                weights.append(w.tolist())
        return weights
    
    def get_activations(self, X):
        """Get layer activations"""
        X = np.array(X, dtype=np.float32)
        activations = []
        
        for layer in self.model.layers:
            if isinstance(layer, layers.Dense):
                intermediate_model = models.Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )
                activation = intermediate_model.predict(X, verbose=0)
                activations.append(activation.tolist())
        
        return activations

# ====================
# PYTORCH MODELS
# ====================

class PyTorchNetwork(nn.Module):
    def __init__(self, config):
        super(PyTorchNetwork, self).__init__()
        self.config = config
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'epoch': []}
        self.framework = 'pytorch'
        
        layers_config = config['layers']
        activation = config.get('activation', 'relu')
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        for i in range(len(layers_config) - 1):
            self.layers.append(nn.Linear(layers_config[i], layers_config[i + 1]))
            
            # Add batch norm if specified
            if config.get('batch_norm', False) and i < len(layers_config) - 2:
                self.layers.append(nn.BatchNorm1d(layers_config[i + 1]))
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(config.get('dropout', 0))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass"""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if not isinstance(layer, nn.BatchNorm1d):
                x = self.activation(x)
                if self.config.get('dropout', 0) > 0:
                    x = self.dropout(x)
        
        # Output layer
        x = self.layers[-1](x)
        x = self.sigmoid(x)
        
        return x
    
    def train_model(self, X, y, epochs=10, validation_split=0.2):
        """Train the model"""
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # Split data
        dataset_size = len(X)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size
        
        indices = torch.randperm(dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Setup optimizer
        optimizer_name = self.config.get('optimizer', 'adam')
        learning_rate = self.config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == y_train).float().mean()
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_predicted = (val_outputs > 0.5).float()
                val_accuracy = (val_predicted == y_val).float().mean()
            
            self.history['loss'].append(float(loss.item()))
            self.history['accuracy'].append(float(accuracy.item() * 100))
            self.history['val_loss'].append(float(val_loss.item()))
            self.history['val_accuracy'].append(float(val_accuracy.item() * 100))
        
        self.history['epoch'].extend(range(len(self.history['epoch']) + 1,
                                          len(self.history['epoch']) + epochs + 1))
        
        return {
            'loss': float(loss.item()),
            'accuracy': float(accuracy.item() * 100),
            'val_loss': float(val_loss.item()),
            'val_accuracy': float(val_accuracy.item() * 100)
        }
    
    def predict_model(self, X):
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X)
            predictions = self.forward(X)
        return predictions.numpy().tolist()
    
    def get_layer_weights(self):
        """Get layer weights"""
        weights = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                w = layer.weight.data.cpu().numpy()
                weights.append(w.T.tolist())
        return weights

# ====================
# SCIKIT-LEARN MODELS
# ====================

class ScikitLearnNetwork:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = {'loss': [], 'accuracy': [], 'epoch': []}
        self.framework = 'sklearn'
        self.scaler = StandardScaler()
        self.build_model()
    
    def build_model(self):
        """Build scikit-learn MLP"""
        layers_config = self.config['layers'][1:-1]  # Hidden layers only
        activation = self.config.get('activation', 'relu')
        learning_rate = self.config.get('learning_rate', 0.001)
        
        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(layers_config),
            activation=activation,
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=1,  # We'll train iteratively
            warm_start=True,
            random_state=42
        )
    
    def train(self, X, y, epochs=10):
        """Train the model"""
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).ravel()
        
        # Fit scaler on first training
        if not hasattr(self.scaler, 'mean_'):
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        for epoch in range(epochs):
            self.model.fit(X, y)
            
            # Calculate metrics
            predictions = self.model.predict(X)
            accuracy = accuracy_score(y, predictions) * 100
            loss = self.model.loss_
            
            self.history['loss'].append(float(loss))
            self.history['accuracy'].append(float(accuracy))
        
        self.history['epoch'].extend(range(len(self.history['epoch']) + 1,
                                          len(self.history['epoch']) + epochs + 1))
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X, dtype=np.float32)
        X = self.scaler.transform(X)
        predictions = self.model.predict_proba(X)
        return predictions.tolist()

# ====================
# DATASETS
# ====================

DATASETS = {
    'xor': {
        'name': 'XOR Problem',
        'X': [[0, 0], [0, 1], [1, 0], [1, 1]],
        'y': [[0], [1], [1], [0]],
        'type': 'classification'
    },
    'and': {
        'name': 'AND Gate',
        'X': [[0, 0], [0, 1], [1, 0], [1, 1]],
        'y': [[0], [0], [0], [1]],
        'type': 'classification'
    },
    'or': {
        'name': 'OR Gate',
        'X': [[0, 0], [0, 1], [1, 0], [1, 1]],
        'y': [[0], [1], [1], [1]],
        'type': 'classification'
    },
    'moons': {
        'name': 'Two Moons',
        'type': 'classification',
        'generator': lambda: make_moons(n_samples=200, noise=0.1, random_state=42)
    },
    'circles': {
        'name': 'Concentric Circles',
        'type': 'classification',
        'generator': lambda: make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
    }
}

def get_dataset(name):
    """Get dataset by name"""
    dataset = DATASETS.get(name)
    if not dataset:
        return None
    
    if 'generator' in dataset:
        X, y = dataset['generator']()
        return {
            'X': X.tolist(),
            'y': y.reshape(-1, 1).tolist(),
            'type': dataset['type']
        }
    
    return {
        'X': dataset['X'],
        'y': dataset['y'],
        'type': dataset['type']
    }

# ====================
# ROUTES
# ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/create', methods=['POST'])
def create_network():
    """Create new neural network"""
    try:
        data = request.json
        framework = data.get('framework', 'tensorflow')
        
        config = {
            'layers': data.get('layers', [2, 4, 1]),
            'activation': data.get('activation', 'relu'),
            'learning_rate': data.get('learning_rate', 0.001),
            'optimizer': data.get('optimizer', 'adam'),
            'dropout': data.get('dropout', 0.0),
            'batch_norm': data.get('batch_norm', False),
            'batch_size': data.get('batch_size', 32)
        }
        
        network_id = str(uuid.uuid4())
        
        if framework == 'tensorflow':
            network = TensorFlowNetwork(config)
        elif framework == 'pytorch':
            network = PyTorchNetwork(config)
        elif framework == 'sklearn':
            network = ScikitLearnNetwork(config)
        else:
            return jsonify({'success': False, 'error': 'Invalid framework'}), 400
        
        networks[network_id] = network
        
        return jsonify({
            'success': True,
            'network_id': network_id,
            'framework': framework,
            'config': config
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/train', methods=['POST'])
def train_network():
    """Train network"""
    try:
        data = request.json
        network_id = data.get('network_id')
        dataset_name = data.get('dataset', 'xor')
        epochs = data.get('epochs', 10)
        
        if network_id not in networks:
            return jsonify({'success': False, 'error': 'Network not found'}), 404
        
        network = networks[network_id]
        dataset = get_dataset(dataset_name)
        
        if not dataset:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        X = dataset['X']
        y = dataset['y']
        
        # Train based on framework
        if network.framework == 'tensorflow':
            result = network.train(X, y, epochs)
        elif network.framework == 'pytorch':
            result = network.train_model(X, y, epochs)
        elif network.framework == 'sklearn':
            result = network.train(X, y, epochs)
        
        return jsonify({
            'success': True,
            'result': result,
            'history': network.history
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction"""
    try:
        data = request.json
        network_id = data.get('network_id')
        input_data = data.get('input')
        
        if network_id not in networks:
            return jsonify({'success': False, 'error': 'Network not found'}), 404
        
        network = networks[network_id]
        
        if network.framework == 'tensorflow':
            predictions = network.predict([input_data])
            activations = network.get_activations([input_data])
            weights = network.get_weights()
        elif network.framework == 'pytorch':
            predictions = network.predict_model([input_data])
            weights = network.get_layer_weights()
            activations = []
        elif network.framework == 'sklearn':
            predictions = network.predict([input_data])
            weights = []
            activations = []
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'activations': activations,
            'weights': weights
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/visualize', methods=['POST'])
def visualize_network():
    """Generate network visualization"""
    try:
        data = request.json
        network_id = data.get('network_id')
        
        if network_id not in networks:
            return jsonify({'success': False, 'error': 'Network not found'}), 404
        
        network = networks[network_id]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Neural Network Analysis - {network.framework.upper()}', fontsize=16)
        
        # Plot 1: Training Loss
        if network.history['loss']:
            axes[0, 0].plot(network.history['epoch'], network.history['loss'], 'b-', label='Training Loss')
            if 'val_loss' in network.history and network.history['val_loss']:
                axes[0, 0].plot(network.history['epoch'], network.history['val_loss'], 'r--', label='Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss Over Time')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        if network.history['accuracy']:
            axes[0, 1].plot(network.history['epoch'], network.history['accuracy'], 'g-', label='Training Accuracy')
            if 'val_accuracy' in network.history and network.history['val_accuracy']:
                axes[0, 1].plot(network.history['epoch'], network.history['val_accuracy'], 'r--', label='Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Accuracy Over Time')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Network Architecture
        layers_text = f"Architecture: {' → '.join(map(str, network.config['layers']))}\n"
        layers_text += f"Activation: {network.config['activation']}\n"
        layers_text += f"Learning Rate: {network.config['learning_rate']}\n"
        layers_text += f"Optimizer: {network.config['optimizer']}"
        
        axes[1, 0].text(0.5, 0.5, layers_text, ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Network Configuration')
        
        # Plot 4: Performance Summary
        if network.history['loss']:
            summary_text = f"Final Loss: {network.history['loss'][-1]:.6f}\n"
            summary_text += f"Final Accuracy: {network.history['accuracy'][-1]:.2f}%\n"
            summary_text += f"Total Epochs: {len(network.history['epoch'])}\n"
            summary_text += f"Framework: {network.framework.upper()}"
            
            axes[1, 1].text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save to bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
        img_bytes.seek(0)
        plt.close()
        
        # Encode to base64
        img_base64 = base64.b64encode(img_bytes.read()).decode()
        
        return jsonify({
            'success': True,
            'image': img_base64
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get available datasets"""
    return jsonify({
        'success': True,
        'datasets': {k: v['name'] for k, v in DATASETS.items()}
    })

@app.route('/api/save', methods=['POST'])
def save_model():
    """Save model"""
    try:
        data = request.json
        network_id = data.get('network_id')
        
        if network_id not in networks:
            return jsonify({'success': False, 'error': 'Network not found'}), 404
        
        network = networks[network_id]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if network.framework == 'tensorflow':
            filename = f'tf_model_{timestamp}.h5'
            filepath = os.path.join('models', filename)
            os.makedirs('models', exist_ok=True)
            network.model.save(filepath)
        elif network.framework == 'pytorch':
            filename = f'torch_model_{timestamp}.pth'
            filepath = os.path.join('models', filename)
            os.makedirs('models', exist_ok=True)
            torch.save(network.state_dict(), filepath)
        elif network.framework == 'sklearn':
            import joblib
            filename = f'sklearn_model_{timestamp}.pkl'
            filepath = os.path.join('models', filename)
            os.makedirs('models', exist_ok=True)
            joblib.dump(network.model, filepath)
        
        return jsonify({
            'success': True,
            'filename': filename
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5010)