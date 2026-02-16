# 🧠 Java Neural Network (From Scratch)

![Java](https://img.shields.io/badge/Java-17%2B-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white)
![Data](https://img.shields.io/badge/Data-CSV-blue?style=for-the-badge)
![Architecture](https://img.shields.io/badge/Architecture-MLP-purple?style=for-the-badge)

A fully connected, 4-layer Neural Network built entirely in Java without any external Machine Learning libraries. This project manually implements **Stochastic Gradient Descent**, **Backpropagation**, and **Matrix Operations** using standard Java Collections.

## 📖 Overview

This engine is designed to classify handwritten digits (0-9) using the CSV version of the MNIST dataset (commonly found on Kaggle).

### 🏭 The Analogy: The Decision Factory
To understand how this code works, imagine a factory designed to identify what is drawn on a piece of paper.

1.  **Input Zone (Layer 0 - 784 Neurons):** The paper is 28x28 pixels. We have 784 workers standing at the entrance. Each worker holds exactly one pixel and shouts out how dark it is (0 to 1).
2.  **Processing Floor 1 (Layer 1 - 128 Neurons):** These workers listen to the shouts from the entrance. They look for specific combinations—like a straight line or a curve. If they hear the right combination, they get excited (activate) and shout to the next floor.
3.  **Processing Floor 2 (Layer 2 - 64 Neurons):** These workers combine the simple patterns from Floor 1 into complex shapes (like loops or angles).
4.  **The Judges (Output Layer - 10 Neurons):** The final 10 judges represent the numbers 0-9. They listen to Floor 2. If the "loop" and "tail" detectors are shouting, the Judge for "9" raises their sign.

## 🏗️ Technical Architecture

Unlike many tutorials that use 1 hidden layer, this engine uses a **Deep Neural Network** architecture defined in `NeuralEngine.java`:

* **Input Layer:** 784 Neurons (Pixel intensities normalized 0.0 - 1.0)
* **Hidden Layer 1:** 128 Neurons
* **Hidden Layer 2:** 64 Neurons
* **Output Layer:** 10 Neurons (Probability score for digits 0-9)
* **Activation:** Sigmoid Function
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Cost Function:** Mean Squared Error (MSE)

## ✨ Features implemented in Raw Java

* **Custom CSV Parser:** `ExcelParse.java` reads Kaggle-style CSVs (`label, pixel0, pixel1...`) without using Apache POI or OpenCSV.
* **Binary Model Persistence:** The `saveModel` and `loadModel` methods use `DataOutputStream` to save weights and biases to a `.bin` file, allowing you to pause training and resume testing later.
* **3D Weight Management:** Weights are stored in a `List<List<List<Float>>>` structure in `Weights.java`, providing a granular view of connections between every single neuron.
* **Mini-Batch Gradient Descent:** The network trains in batches of 1000 images (`BATCH_SIZE`) to stabilize the learning process.

## 🚀 Getting Started

### 1. Prerequisites
* Java Development Kit (JDK) 8 or higher.
* **The Dataset:** This code uses the CSV format (Kaggle Digit Recognizer).
    * Download `train.csv` from [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data).
    * Place it in a folder named `digit-recognizer` in your project root.

### 2. Project Setup
Ensure your directory structure looks like this:

```text
ProjectRoot/
├── digit-recognizer/
│   └── train.csv          <-- Download this!
├── src/
│   ├── NeuralEngine.java
│   ├── NeuralNetwork.java
│   ├── Neuron.java
│   ├── Weights.java
│   ├── ExcelParse.java
│   └── main.java
```

### 3. Class Descriptions

1. **Neuron** Represents a single unit. Handles the Sigmoid activation logic and stores the bias.
2. **Weights** A wrapper around a 3D List List<List<List<Float>>>. Manages the connection strength between every neuron in Layer A and Layer B.
3. **NeuralNetwork** The data structure holding the List of Neurons and the Weights object.
4. **NeuralEngine** "The Brain". It contains the hyperparameters (LEARNING_RATE = 0.01, BATCH_SIZE = 1000). Handles forwardPass(), backpropagate(), and train().
5. **ExcelParse** A custom utility to read the CSV file, split the label from the pixels, and normalize pixel values by dividing by 255.


### 4. How to Use

The main.java file contains two modes. You can uncomment the one you need.

* To Train a New Model:


    // In main.java
    public static void main(String[] args) throws Exception {
    createModel(); // Trains for 50 epochs and saves to 'model.bin'
    }


* To Test an Existing Model:


    // In main.java
    public static void main(String[] args) throws Exception {
    // Loads 'model.bin' and runs accuracy tests
    NeuralEngine testEngine = NeuralEngine.loadModel("model.bin");
    
        // Loads data to test against
        float[][] data = ExcelParse.loadData("digit-recognizer/train.csv", 40000);
        testEngine.test(data);
    }




## 📊 Performance

* Training Speed: On an M3 MacBook Pro, 50 epochs take approximately 7 minutes (~425 seconds).

* Accuracy: With the current architecture (784-128-64-10) and learning rate (0.01), accuracy typically converges around 85-95% depending on the structure of the network.