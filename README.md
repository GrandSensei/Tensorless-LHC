# Particle Shower Classification with Neural Networks

A machine learning project for classifying particle types (electrons, pions, muons, and gamma rays) based on calorimeter shower patterns from Geant4 simulations.

## 🎯 Project Overview

This project implements a **custom neural network from scratch in Java** to classify particle types based on energy deposition patterns in a sampling calorimeter. The work combines high-energy physics simulation (Geant4) with machine learning to achieve ~78-82% classification accuracy on simulated detector data.

### Key Features
- ✅ Custom neural network implementation (no external ML libraries)
- ✅ Physics-informed feature engineering
- ✅ ReLU + Softmax architecture with L2 regularization
- ✅ Cross-entropy loss with learning rate decay
- ✅ Complete training pipeline from simulation to inference

---

## 🔬 Physics Background

### The Problem: Particle Identification

In high-energy physics experiments, accurately identifying particle types is crucial for:
- Event reconstruction in detector systems
- Background rejection in physics analyses
- Trigger decisions in real-time data acquisition

## ⚖️ Design Decisions: Why Copper?

A critical part of this project was optimizing the absorber material to maximize classification accuracy within the constraints of a 10-layer sampling calorimeter.

### The Trade-off: Lead (Pb) vs. Copper (Cu)

We analyzed two standard absorber materials:

1.  **Lead (Pb)**: High density ($11.3$ g/cm³) and short Radiation Length ($X_0 \approx 0.56$ cm).
  * *Issue:* With 10mm plates, Lead is "too effective." It converts Gamma rays into showers almost immediately. This makes **Gammas look identical to Electrons** in the first layer, dropping classification precision.
2.  **Copper (Cu)**: Moderate density ($8.9$ g/cm³) and longer Radiation Length ($X_0 \approx 1.43$ cm).
  * *Benefit:* It is "transparent" enough to let Gammas pass the first layer (preserving the unique $\gamma$ signature) but dense enough to force Pions to shower (distinguishing them from Muons).

**Current Choice:** We selected **Copper** as the optimal material for the current fixed geometry (10mm plates), achieving the highest balanced accuracy (~83%).

### Future Roadmap: Moving to Lead
While Copper works best for this specific geometry, **Lead** remains the industry standard for high-performance electromagnetic calorimeters (e.g., CERN CMS ECAL) due to its superior containment.

**Plan for v2.0:**
* **Switch to Lead Absorbers:** To minimize leakage and improve energy resolution.
* **Geometry Overhaul:** Reduce plate thickness from **10mm $\to$ 2mm** and increase layer count from **10 $\to$ 40**.
* **Physics Goal:** This finer segmentation will allow us to use Lead's superior stopping power without losing the "early shower" information required to separate Electrons from Gammas.


### Detector Setup

**Sampling Calorimeter** (Geant4 B4a example):
- **10 layers** of alternating absorber/active material
- **Absorber**: 10mm Copper (Cu) - where particles shower
- **Active Material**: 5mm Liquid Argon - where energy is measured
- **Dimensions**: 10cm × 10cm × 15cm total depth

### Particle Signatures

Different particles create distinct shower patterns:

| Particle | PDG Code | Signature | Key Features |
|----------|----------|-----------|--------------|
| **Electron (e⁻)** | 11 | Electromagnetic shower starting immediately | High energy in layers 0-2, rapid decay |
| **Pion (π⁻)** | 211 | Hadronic shower, more spread out | Variable pattern, late energy deposits |
| **Muon (μ⁻)** | 13 | Minimum ionizing particle | Uniform ~12-15 MeV across all layers |
| **Gamma (γ)** | 22 | Electromagnetic shower starting deeper | Often zero in layer 0, peak at layers 3-5 |

---

## 🏗️ Architecture

### Neural Network Design

```
Input Layer (16 features)
    ↓
Hidden Layer 1 (32 neurons, ReLU)
    ↓
Hidden Layer 2 (16 neurons, ReLU)
    ↓
Output Layer (4 neurons, Softmax)
```

**Key Components:**
- **Activation Functions**: ReLU (hidden) for non-linearity, Softmax (output) for probability distribution
- **Optimization**: Stochastic Gradient Descent with momentum (batch size: 32)
- **Regularization**: L2 weight decay (λ=0.0001) to prevent overfitting
- **Learning Rate**: 0.02-0.05 with exponential decay (×0.9 every 1000 epochs)
- **Weight Initialization**: Xavier/He initialization for stable training

### Feature Engineering (16 Total Features)

#### Raw Features (10)
- **Log-scaled layer energies**: `log(1 + E_i)` for layers 0-9
    - Log scaling handles wide dynamic range (0-200 MeV)
    - Min-max normalization to [0, 1]

#### Engineered Features (6)
1. **Total Energy (Z-scored)**: `ΣE_i` - distinguishes high vs. low energy events
2. **Layer 0 Ratio (Z-scored)**: `E_0 / Total` - key for gamma/electron separation
3. **Peak Position (Z-scored)**: `argmax(E_i) / 9` - shower maximum location
4. **Early Fraction**: `(E_0 + E_1 + E_2) / Total` - early vs. late shower development
5. **Layer 0 Interaction (Binary)**: `E_0 > 0.1 ? 1 : 0` - gamma conversion indicator
6. **Roughness (StdDev)**: `σ(E_i) / 100` - shower shape uniformity (pion/muon separator)

**Physics Motivation:**
- Electrons/gammas shower electromagnetically → distinct energy profiles
- Muons are minimum ionizing → uniform energy across layers
- Pions interact hadronically → higher variance in energy distribution

---

## 📁 Project Structure

```
Geant4XTensorLess/
│
├── java_core/                      # Neural network implementation
│   ├── NeuralEngine.java           # Training, inference, model persistence
│   ├── NeuralNetwork.java          # Network structure
│   ├── Neuron.java                 # Neuron with ReLU/Softmax
│   ├── ExcelParse.java             # Data loading & feature engineering
│   ├── PhysicsClassifier.java      # Main training script
│   └── data/
│       ├── training_data_Cu2.csv   # Simulated particle events
│       └── multiclass_classifier2.bin   
│
└── geant4_DataResource/            # Geant4 B4a modified example
    ├── CMakeLists.txt
    └── src/
        ├── DetectorConstruction.cc # Calorimeter geometry (Cu + LAr)
        ├── EventAction.cc          # Energy collection per layer
        └── PrimaryGeneratorAction.cc

```

---

## 🚀 Getting Started

### Prerequisites

**For Simulation (Optional as pre-generated data included):**
- Geant4 11.0+ ([installation guide](https://geant4.web.cern.ch/support/download))
- CMake 3.16+
- C++17 compiler

**For Training & Inference:**
- Java 17+ (OpenJDK recommended)
- No external ML libraries required

### Running the Project

#### 1. Generate Training Data (Optional)

```bash
cd geant4_DataResource/build
cmake ..
make
./exampleB4a run1.mac
# Output: training_data.csv with particle shower events in the cmake-build-debug
```

The simulation fires particles at the calorimeter and records energy deposits per layer.

#### 2. Train the Neural Network

```bash
cd java_core
javac *.java
java PhysicsClassifier
```

**Training Output:**
```
Loading Multi-Class Physics Data...
Reading file with Enhanced Physics Features: data/training_data_Cu2.csv
Feature Stats:
  Ratio - Mean: 0.088917404 Std: 0.1080856
  Total - Mean: 198.18878 Std: 65.41866
  Peak  - Mean: 0.45594794 Std: 0.28404868
Loaded 79999 events with 16 features (10 layers + 6 engineered)
Loaded 79999 events.
Shuffling data...
Training set: 63999 samples
Test set: 16000 samples

--- TRAINING DATA INSPECTION ---
CLASS DISTRIBUTION:
  Electron (label=0): 14396 (22.5%)
  Pion (label=1): 16062 (25.1%)
  Muon (label=2): 15977 (25.0%)
  Gamma (label=3): 17564 (27.4%)

Class imbalance: 18.0% ⚠ Consider balancing

Sample features (first 3 of each class):
Electron: L0=49.3 L1=55.1 L2=69.0 ... Ratio=-0.295 EarlyFrac=0.322, 1.000,0.249
Pion: L0=70.8 L1=2.0 L2=0.0 ... Ratio=8.272 EarlyFrac=0.985, 1.000,0.173
Muon: L0=44.4 L1=48.4 L2=45.1 ... Ratio=-0.031 EarlyFrac=0.285, 1.000,0.015
Gamma: L0=58.0 L1=62.8 L2=59.7 ... Ratio=0.145 EarlyFrac=0.360, 1.000,0.158
....
....
....

------------------------------

 Engine Initialized: 17 In -> 4 Out
Starting Training...
Architecture: 16 -> 64 -> 32 -> 4
Activation: ReLU (hidden) + Softmax (output)
Loss: Cross-Entropy with L2 Regularization
----------------------------------------

Epoch 0 | Cost: 0.472512 | LR: 0.020000
Epoch 100 | Cost: 0.397970 | LR: 0.020000
....
....
....
------------------------------------------------
EARLY STOPPING TRIGGERED
Cost hasn't improved in 500 epochs. We are done.
Best Cost: 0.39013854
------------------------------------------------


========================================
         FINAL EVALUATION
========================================

--- TEST SET (Unseen Data) ---
----------------------------------
Running Multi-Class Test (Geant4 Labels)...

--- CONFUSION MATRIX ---
Act \ Pred | Elec  | Pion  | Muon  | Gamma | 
Elec       | 3440   | 55     | 3      | 105    | 
Pion       | 69     | 2452   | 1407   | 10     | 
Muon       | 1      | 36     | 3986   | 0      | 
Gamma      | 1096   | 19     | 0      | 3321   | 

FINAL ACCURACY: 82.49%
----------------------------------

--- TRAINING SET (For Comparison) ---
----------------------------------
Running Multi-Class Test (Geant4 Labels)...

--- CONFUSION MATRIX ---
Act \ Pred | Elec  | Pion  | Muon  | Gamma | 
Elec       | 13754  | 226    | 6      | 410    | 
Pion       | 293    | 10145  | 5595   | 29     | 
Muon       | 0      | 102    | 15875  | 0      | 
Gamma      | 4376   | 86     | 2      | 13100  | 

FINAL ACCURACY: 82.62%
----------------------------------
Saving model to multiclass_classifier2.bin
Model saved!!

========================================
         TRAINING COMPLETE
========================================

OVERFITTING ANALYSIS:
  Train Accuracy: 82.62%
  Test Accuracy:  82.49%
  Gap:            0.12%
  Status: ✓ Excellent generalization
```

#### 3. Test the Model

```java
// In PhysicsClassifier.java
NeuralEngine brain = NeuralEngine.loadModel("multiclass_classifier.bin");
brain.test(testData);
```

---

## 📊 Results
**Quick Stats:**
- 🎯 Overall Accuracy: 82.5%
- 📈 Generalization Gap: 0.12% (Excellent!)
- 🔬 Dataset: 80k events (64k train / 16k test)
- ⚡ Training Time: ~10 minutes
- 🎨 No overfitting observed

### Performance Metrics

| Metric | Training Set | Test Set | Status |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | 82.6% | 82.5% | Consistent (No Overfitting) |
| **Electron Precision** | 0.75 | 0.75 | Low due to Gamma contamination |
| **Pion Precision** | 0.96 | 0.96 | High (Very few things fake being a Pion) |
| **Muon Precision** | 0.74 | 0.74 | Low due to Pion "punch-through" |
| **Gamma Precision** | 0.97 | 0.97 | High (Model is confident when it calls Gamma) |
**Generalization Gap**: 0.1% (indicates good generalization, minimal overfitting)

### Confusion Matrix (Test Set)

```
### Confusion Matrix (Test Set - 16,000 Events)

| Act \ Pred | Electron | Pion | Muon | Gamma |
|------------|----------|------|------|-------|
| **Electron** | **3,440** | 55 | 3 | 105 |
| **Pion** | 69 | **2,452** | 1,407 | 10 |
| **Muon** | 1 | 36 | **3,986** | 0 |
| **Gamma** | 1,096 | 19 | 0 | **3,321** |

*Note: The high error rate between Pions and Muons (1,407 misclassified) represents the physical "punch-through" limit of the detector geometry.*
```
**Performance Context:**
- **82.5% accuracy** on 4-class classification is strong, especially considering:
  - Pion/Muon confusion is a **physical detector limitation** (hadronic punch-through)
  - Gamma/Electron overlap is expected in calorimetry (both are EM showers)
- The **0.12% train-test gap** proves the model learned physics, not memorization

### Training Characteristics

- **Convergence**: ~2000-3000 epochs
- **Training Time**: ~5-10 minutes on standard laptop
- **Model Size**: 3.2 KB (2,564 parameters)
- **Best Learning Rate**: 0.02-0.05 (dataset-dependent)

---

## 🔧 Technical Implementation Details

### Custom Neural Network Features

#### Forward Propagation with Softmax
A bunch of psuedocode for reference
```java
public void forwardPass() {
    // Hidden layer computation with ReLU
    for (layer in hiddenLayers) {
        for (neuron in layer) {
            sum = bias + Σ(weight_ij × activation_i)
            activation = max(0, sum)  // ReLU
        }
    }
    
    // Output layer with numerically stable softmax
    maxVal = max(outputNeurons)
    for (neuron in outputLayer) {
        exp_vals[i] = exp(neuron.val - maxVal)
    }
    softmax[i] = exp_vals[i] / Σ(exp_vals)
}
```

#### Backpropagation with L2 Regularization
```java
public void backpropagate(targets, learningRate) {
    // Output layer: Softmax + Cross-Entropy derivative
    error = prediction - target  // Clean derivative!
    
    // Hidden layers: Chain rule + ReLU derivative
    for (layer in reverse(hiddenLayers)) {
        gradient = error × activation
        gradient = clip(gradient, -5, 5)  // Prevent explosions
        
        // L2 regularization (weight decay)
        weight_update = -LR × (gradient + λ × weight)
    }
}
```

#### Early Stopping & Learning Rate Decay
```java
if (cost < bestCost - threshold) {
    bestCost = cost
    patience = 0
} else {
    patience++
}

if (patience > 500) stopTraining()
if (epoch % 1000 == 0) learningRate *= 0.9
```

### Data Pipeline

1. **Simulation** (Geant4): 300 MeV particles → 10-layer calorimeter
2. **Preprocessing**: Log scaling, Z-score normalization
3. **Feature Engineering**: Physics-informed calculations
4. **Training**: Mini-batch SGD (32 samples)
5. **Validation**: 80/20 train-test split
6. **Persistence**: Binary model serialization

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

### Machine Learning
- Neural network implementation from first principles
- Backpropagation and gradient descent
- Regularization techniques (L2, early stopping)
- Hyperparameter optimization
- Train/test methodology

### Physics
- Particle detector principles
- Calorimetry and energy measurement
- Particle shower phenomenology
- Monte Carlo simulation (Geant4)

### Software Engineering
- Object-oriented design (Java)
- Model serialization and deployment
- Data pipeline construction
- Performance optimization

---

## 🚧 Future Improvements

### Model Enhancements
- [ ] Implement dropout for better regularization
- [ ] Try deeper architectures (4-5 hidden layers)
- [ ] Experiment with batch normalization
- [ ] Add momentum/Adam optimizer

### Feature Engineering
- [ ] Shower shape moments (skewness, kurtosis)
- [ ] Energy asymmetry measures
- [ ] Layer-to-layer transition features
- [ ] Interaction point estimation

### Data & Validation
- [ ] Generate larger dataset (100k+ events)
- [ ] K-fold cross-validation
- [ ] Test on different detector materials (Pb, Fe)
- [ ] Real experimental data validation (if available)

### Physics Extensions
- [ ] Add more particle types (protons, kaons)
- [ ] Higher-energy training (100 GeV - 1 TeV)
- [ ] Pile-up simulation (multiple particles)

---

## 📚 References

### Geant4 Simulation
- [Geant4 Collaboration](https://geant4.web.cern.ch/)
- [B4 Example Documentation](https://geant4-userdoc.web.cern.ch/Examples/basic.html#b4)

### Physics Background
- [Particle Data Group (PDG)](https://pdg.lbl.gov/) - Passage on Calorimetry
- Fabrici & Gianotti, *Calorimetry for Particle Physics* (Rev. Mod. Phys. 75, 1243)

### Machine Learning
- Goodfellow et al., *Deep Learning* (2016), Chapter 6 (feedforward networks)
- Nielsen, *Neural Networks and Deep Learning* (online book)

---

## 🤝 Contributing & Contact

This project was developed to demonstrate the intersection of High-Energy Physics (HEP) and software engineering, specifically focusing on how low-level algorithms can be optimized for physics data analysis.

**Author**: Mustafa Bazi

**Purpose**: Demonstrate machine learning and high-energy physics skills  

**Status**: Active development (as of January 2025)

### Acknowledgments
- Geant4 Collaboration for the excellent simulation toolkit. It is one of the coolest pieces of simulators I have worked with.

---

## 📄 License

This project uses:
- Geant4 (Geant4 Software License)
- Custom code (MIT License)

For educational and research purposes.

---

## 🏆 Project Highlights for Resume/CV

**Key Achievement**: Built a particle classifier from scratch achieving **83% accuracy** on simulated calorimeter data, demonstrating understanding of both neural network fundamentals and particle physics.

**Technical Skills Showcased**:
- Monte Carlo simulation (Geant4/C++)
- Neural network implementation (Java)
- Physics-informed feature engineering
- Scientific programming and data analysis

**Relevant for**: Particle physics, detector R&D, machine learning engineering, data science roles in HEP

---

*"Learning by doing: Building a neural network from scratch teaches you more than using PyTorch ever could."*