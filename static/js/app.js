// Advanced Neural Network Simulator - Professional Edition
// Complete JavaScript Implementation

// Global State
let appState = {
    networkId: null,
    framework: 'tensorflow',
    isTraining: false,
    config: null,
    history: null
};

// Charts
let lossChart = null;
let accuracyChart = null;

// DOM Elements
const elements = {
    // Framework Selection
    frameworkOptions: document.querySelectorAll('.framework-option'),
    frameworkBadge: document.getElementById('framework-badge'),
    
    // Architecture Configuration
    inputLayer: document.getElementById('input-layer'),
    hiddenLayers: document.getElementById('hidden-layers'),
    outputLayer: document.getElementById('output-layer'),
    activation: document.getElementById('activation'),
    optimizer: document.getElementById('optimizer'),
    learningRate: document.getElementById('learning-rate'),
    batchSize: document.getElementById('batch-size'),
    dropout: document.getElementById('dropout'),
    batchNorm: document.getElementById('batch-norm'),
    
    // Network Controls
    createBtn: document.getElementById('create-network'),
    trainBtn: document.getElementById('train-btn'),
    stopBtn: document.getElementById('stop-btn'),
    resetBtn: document.getElementById('reset-btn'),
    
    // Training Configuration
    dataset: document.getElementById('dataset'),
    epochs: document.getElementById('epochs'),
    
    // Metrics
    trainLoss: document.getElementById('train-loss'),
    trainAccuracy: document.getElementById('train-accuracy'),
    valAccuracy: document.getElementById('val-accuracy'),
    totalEpochs: document.getElementById('total-epochs'),
    
    // Visualization
    networkCanvas: document.getElementById('network-canvas'),
    canvasPlaceholder: document.getElementById('canvas-placeholder'),
    generateViz: document.getElementById('generate-viz'),
    vizPreview: document.getElementById('viz-preview'),
    
    // Prediction
    testInput: document.getElementById('test-input'),
    predictBtn: document.getElementById('predict-btn'),
    predictionResult: document.getElementById('prediction-result'),
    
    // Charts
    lossPlaceholder: document.getElementById('loss-placeholder'),
    accPlaceholder: document.getElementById('acc-placeholder'),
    
    // Loading
    loadingOverlay: document.getElementById('loading-overlay'),
    
    // Export & About
    downloadReport: document.getElementById('download-report'),
    aboutBtn: document.getElementById('about-btn'),
    aboutModal: document.getElementById('about-modal'),
    closeAbout: document.getElementById('close-about')
};

// Initialize Application
function initApp() {
    setupEventListeners();
    initCharts();
    setupCanvas();
}

// Setup Event Listeners
function setupEventListeners() {
    // Framework Selection
    elements.frameworkOptions.forEach(option => {
        option.addEventListener('click', () => {
            selectFramework(option.dataset.framework);
        });
    });
    
    // Network Creation
    elements.createBtn.addEventListener('click', createNetwork);
    
    // Training Controls
    elements.trainBtn.addEventListener('click', trainNetwork);
    elements.stopBtn.addEventListener('click', stopTraining);
    elements.resetBtn.addEventListener('click', resetNetwork);
    
    // Prediction
    elements.predictBtn.addEventListener('click', makePrediction);
    
    // Visualization
    elements.generateViz.addEventListener('click', generateVisualization);
    
    // Export
    elements.downloadReport.addEventListener('click', downloadReport);
    
    // About Modal
    if (elements.aboutBtn) {
        elements.aboutBtn.addEventListener('click', () => {
            elements.aboutModal.classList.add('active');
        });
    }
    
    if (elements.closeAbout) {
        elements.closeAbout.addEventListener('click', () => {
            elements.aboutModal.classList.remove('active');
        });
    }
    
    if (elements.aboutModal) {
        elements.aboutModal.addEventListener('click', (e) => {
            if (e.target === elements.aboutModal) {
                elements.aboutModal.classList.remove('active');
            }
        });
    }
}

// Framework Selection
function selectFramework(framework) {
    appState.framework = framework;
    
    elements.frameworkOptions.forEach(opt => opt.classList.remove('active'));
    document.querySelector(`[data-framework="${framework}"]`).classList.add('active');
    
    const frameworkNames = {
        'tensorflow': 'TensorFlow',
        'pytorch': 'PyTorch',
        'sklearn': 'Scikit-learn'
    };
    
    elements.frameworkBadge.textContent = frameworkNames[framework];
    
    showNotification(`Switched to ${frameworkNames[framework]}`, 'info');
}

// Initialize Charts
function initCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 300
        },
        plugins: {
            legend: {
                display: true,
                labels: {
                    color: '#d1d5db',
                    font: {
                        family: 'Outfit',
                        size: 11
                    }
                }
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)',
                    drawBorder: false
                },
                ticks: {
                    color: '#9ca3af',
                    font: {
                        family: 'Fira Code',
                        size: 10
                    }
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)',
                    drawBorder: false
                },
                ticks: {
                    color: '#9ca3af',
                    font: {
                        family: 'Fira Code',
                        size: 10
                    }
                }
            }
        }
    };
    
    // Loss Chart
    lossChart = new Chart(document.getElementById('loss-chart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: chartOptions
    });
    
    // Accuracy Chart
    accuracyChart = new Chart(document.getElementById('accuracy-chart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Validation Accuracy',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            ...chartOptions,
            scales: {
                ...chartOptions.scales,
                y: {
                    ...chartOptions.scales.y,
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

// Update Charts
function updateCharts(history) {
    if (!history || !history.epoch || history.epoch.length === 0) return;
    
    elements.lossPlaceholder.classList.add('hidden');
    elements.accPlaceholder.classList.add('hidden');
    
    // Update Loss Chart
    lossChart.data.labels = history.epoch;
    lossChart.data.datasets[0].data = history.loss;
    lossChart.data.datasets[1].data = history.val_loss || [];
    lossChart.update();
    
    // Update Accuracy Chart
    accuracyChart.data.labels = history.epoch;
    accuracyChart.data.datasets[0].data = history.accuracy;
    accuracyChart.data.datasets[1].data = history.val_accuracy || [];
    accuracyChart.update();
}

// Setup Canvas
function setupCanvas() {
    const canvas = elements.networkCanvas;
    const container = canvas.parentElement;
    
    canvas.width = container.clientWidth;
    canvas.height = 400;
    
    window.addEventListener('resize', () => {
        canvas.width = container.clientWidth;
        canvas.height = 400;
        if (appState.networkId) {
            drawNetwork();
        }
    });
}

// Draw Network on Canvas
function drawNetwork() {
    if (!appState.config) return;
    
    const canvas = elements.networkCanvas;
    const ctx = canvas.getContext('2d');
    const layers = appState.config.layers;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const padding = 60;
    const maxNeurons = Math.max(...layers);
    const layerSpacing = (canvas.width - padding * 2) / (layers.length - 1);
    
    // Calculate neuron positions
    const positions = [];
    for (let i = 0; i < layers.length; i++) {
        const layerNeurons = layers[i];
        const neuronSpacing = (canvas.height - padding * 2) / (maxNeurons + 1);
        const offset = (maxNeurons - layerNeurons) / 2;
        
        const neurons = [];
        for (let j = 0; j < layerNeurons; j++) {
            neurons.push({
                x: padding + i * layerSpacing,
                y: padding + (offset + j + 1) * neuronSpacing
            });
        }
        positions.push(neurons);
    }
    
    // Draw connections
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.3)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i < positions.length - 1; i++) {
        for (let j = 0; j < positions[i].length; j++) {
            for (let k = 0; k < positions[i + 1].length; k++) {
                ctx.beginPath();
                ctx.moveTo(positions[i][j].x, positions[i][j].y);
                ctx.lineTo(positions[i + 1][k].x, positions[i + 1][k].y);
                ctx.stroke();
            }
        }
    }
    
    // Draw neurons
    positions.forEach((layer, layerIdx) => {
        layer.forEach((neuron, neuronIdx) => {
            // Neuron circle
            ctx.beginPath();
            ctx.arc(neuron.x, neuron.y, 15, 0, Math.PI * 2);
            
            // Gradient fill
            const gradient = ctx.createRadialGradient(neuron.x, neuron.y, 0, neuron.x, neuron.y, 15);
            gradient.addColorStop(0, '#3385ff');
            gradient.addColorStop(1, '#0066ff');
            ctx.fillStyle = gradient;
            ctx.fill();
            
            // Border
            ctx.strokeStyle = '#00d4ff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Glow effect
            ctx.shadowColor = '#00d4ff';
            ctx.shadowBlur = 10;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 0;
        });
    });
    
    // Reset shadow
    ctx.shadowBlur = 0;
    
    // Draw labels
    ctx.fillStyle = '#f9fafb';
    ctx.font = '600 12px Outfit';
    ctx.textAlign = 'center';
    
    const labels = ['Input', ...Array(layers.length - 2).fill('').map((_, i) => `Hidden ${i + 1}`), 'Output'];
    labels.forEach((label, i) => {
        if (positions[i] && positions[i][0]) {
            ctx.fillText(label, positions[i][0].x, 30);
        }
    });
}

// API Calls
async function apiCall(endpoint, data = {}) {
    try {
        showLoading();
        const response = await fetch(`/api/${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        hideLoading();
        
        if (!result.success) {
            throw new Error(result.error || 'Request failed');
        }
        
        return result;
    } catch (error) {
        hideLoading();
        showNotification(error.message, 'error');
        throw error;
    }
}

// Create Network
async function createNetwork() {
    try {
        const layers = [];
        
        // Input layer
        layers.push(parseInt(elements.inputLayer.value));
        
        // Hidden layers
        const hiddenInput = elements.hiddenLayers.value.trim();
        if (hiddenInput) {
            const hiddenLayers = hiddenInput.split(',')
                .map(n => parseInt(n.trim()))
                .filter(n => !isNaN(n) && n > 0);
            layers.push(...hiddenLayers);
        }
        
        // Output layer
        layers.push(parseInt(elements.outputLayer.value));
        
        // Validate
        if (layers.some(l => isNaN(l) || l <= 0)) {
            throw new Error('Please enter valid layer sizes');
        }
        
        const config = {
            framework: appState.framework,
            layers: layers,
            activation: elements.activation.value,
            optimizer: elements.optimizer.value,
            learning_rate: parseFloat(elements.learningRate.value),
            batch_size: parseInt(elements.batchSize.value),
            dropout: parseFloat(elements.dropout.value),
            batch_norm: elements.batchNorm.checked
        };
        
        const result = await apiCall('create', config);
        
        appState.networkId = result.network_id;
        appState.config = result.config;
        
        elements.canvasPlaceholder.classList.add('hidden');
        drawNetwork();
        
        enableControls(true);
        showNotification('Neural network created successfully!', 'success');
        
    } catch (error) {
        console.error('Error creating network:', error);
    }
}

// Train Network
async function trainNetwork() {
    if (!appState.networkId) return;
    
    try {
        appState.isTraining = true;
        updateTrainingState(true);
        
        const data = {
            network_id: appState.networkId,
            dataset: elements.dataset.value,
            epochs: parseInt(elements.epochs.value)
        };
        
        const result = await apiCall('train', data);
        
        appState.history = result.history;
        
        // Update metrics
        if (result.result) {
            elements.trainLoss.textContent = result.result.loss.toFixed(6);
            elements.trainAccuracy.textContent = result.result.accuracy.toFixed(2) + '%';
            elements.valAccuracy.textContent = (result.result.val_accuracy || 0).toFixed(2) + '%';
            elements.totalEpochs.textContent = result.history.epoch.length;
        }
        
        // Update charts
        updateCharts(result.history);
        
        showNotification('Training completed!', 'success');
        
    } catch (error) {
        console.error('Error training network:', error);
    } finally {
        appState.isTraining = false;
        updateTrainingState(false);
    }
}

// Stop Training
function stopTraining() {
    appState.isTraining = false;
    updateTrainingState(false);
    showNotification('Training stopped', 'info');
}

// Reset Network
async function resetNetwork() {
    if (!confirm('Reset network? All training progress will be lost.')) {
        return;
    }
    
    appState.networkId = null;
    appState.config = null;
    appState.history = null;
    
    // Reset charts
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.data.datasets[1].data = [];
    lossChart.update();
    
    accuracyChart.data.labels = [];
    accuracyChart.data.datasets[0].data = [];
    accuracyChart.data.datasets[1].data = [];
    accuracyChart.update();
    
    elements.lossPlaceholder.classList.remove('hidden');
    elements.accPlaceholder.classList.remove('hidden');
    
    // Reset metrics
    elements.trainLoss.textContent = '--';
    elements.trainAccuracy.textContent = '--';
    elements.valAccuracy.textContent = '--';
    elements.totalEpochs.textContent = '0';
    
    // Reset canvas
    const ctx = elements.networkCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.networkCanvas.width, elements.networkCanvas.height);
    elements.canvasPlaceholder.classList.remove('hidden');
    
    enableControls(false);
    showNotification('Network reset', 'info');
}

// Make Prediction
async function makePrediction() {
    if (!appState.networkId) return;
    
    try {
        const inputValues = elements.testInput.value
            .split(',')
            .map(v => parseFloat(v.trim()))
            .filter(v => !isNaN(v));
        
        if (inputValues.length !== appState.config.layers[0]) {
            throw new Error(`Expected ${appState.config.layers[0]} input values`);
        }
        
        const result = await apiCall('predict', {
            network_id: appState.networkId,
            input: inputValues
        });
        
        const predictions = result.predictions[0];
        
        elements.predictionResult.innerHTML = `
            <div style="margin-bottom: 0.5rem;">
                <strong style="color: #00d4ff;">Prediction:</strong>
            </div>
            <div style="font-family: var(--font-code); color: #10b981;">
                ${predictions.map((p, i) => `Output ${i + 1}: ${p.toFixed(4)}`).join('<br>')}
            </div>
        `;
        elements.predictionResult.classList.add('show');
        
    } catch (error) {
        console.error('Error making prediction:', error);
    }
}

// Generate Visualization
async function generateVisualization() {
    if (!appState.networkId) return;
    
    try {
        const result = await apiCall('visualize', {
            network_id: appState.networkId
        });
        
        elements.vizPreview.innerHTML = `
            <img src="data:image/png;base64,${result.image}" alt="Network Analysis">
        `;
        elements.vizPreview.classList.add('show');
        
        showNotification('Visualization generated!', 'success');
        
    } catch (error) {
        console.error('Error generating visualization:', error);
    }
}

// Download Report
function downloadReport() {
    showNotification('Report download will be available soon', 'info');
}

// Enable/Disable Controls
function enableControls(enabled) {
    elements.trainBtn.disabled = !enabled;
    elements.resetBtn.disabled = !enabled;
    elements.predictBtn.disabled = !enabled;
    elements.generateViz.disabled = !enabled;
}

// Update Training State
function updateTrainingState(isTraining) {
    elements.trainBtn.disabled = isTraining;
    elements.stopBtn.disabled = !isTraining;
    
    if (isTraining) {
        elements.trainBtn.innerHTML = `
            <svg class="spin" width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 2v3M8 11v3M14 8h-3M5 8H2" stroke="currentColor" stroke-width="2"/>
            </svg>
            Training...
        `;
    } else {
        elements.trainBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M4 3l8 5-8 5V3z" fill="currentColor"/>
            </svg>
            Train Network
        `;
    }
}

// Show Loading
function showLoading() {
    elements.loadingOverlay.classList.add('active');
}

// Hide Loading
function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

// Show Notification
function showNotification(message, type = 'info') {
    const colors = {
        success: '#10b981',
        error: '#ef4444',
        info: '#00d4ff'
    };
    
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: var(--bg-elevated);
        border: 1px solid ${colors[type]};
        border-radius: var(--radius-md);
        color: var(--text-primary);
        box-shadow: var(--shadow-lg), 0 0 20px ${colors[type]}40;
        z-index: 10000;
        animation: slideInRight 0.3s ease;
        font-size: 0.875rem;
        font-weight: 600;
    `;
    
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .spin {
        animation: spin 1s linear infinite;
    }
`;
document.head.appendChild(style);

// Initialize on page load
document.addEventListener('DOMContentLoaded', initApp);