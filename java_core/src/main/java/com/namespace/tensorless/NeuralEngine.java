package com.namespace.tensorless;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class NeuralEngine {



    public NeuralNetwork neuralNetwork;
    private float[][] trainingData;
    private final int BATCH_SIZE= 32;
    private  int LAYERS = 3;
    public int INPUT_SIZE = 784;
    private  int OUTPUT_SIZE =10 ;
    private  float LEARNING_RATE = 0.01f;
    private final int STANDARD =0;

    public NeuralEngine() {
        neuralNetwork = new NeuralNetwork(LAYERS);
        // This creates a 4 layer neural network with one input layer, 2 hidden layers and one output layer.
        setLayer(0,INPUT_SIZE);
        setLayer(1,128);
        setLayer(2,64);
        setLayer(3,OUTPUT_SIZE);
        System.out.println("The neural network has been set");
        setWeights();
        System.out.println("The weights have been set(randomized)");
    }

    public NeuralEngine(int layers) {
        if(layers == STANDARD) setNeuralNetwork(LAYERS);
        else  setNeuralNetwork(layers);
    }


    public NeuralEngine(int inputSize, int outputSize,int layers) {
        this.INPUT_SIZE = inputSize;
        this.OUTPUT_SIZE = outputSize;
        this.LAYERS = layers;
        // We can use smaller hidden layers for this simpler problem
        // 10 Inputs -> 16 Neurons -> 16 Neurons -> 2 Outputs
        neuralNetwork = new NeuralNetwork(LAYERS);
        if (layers == 4) {
            setLayer(0, INPUT_SIZE);
            setLayer(1, 256);
            setLayer(2, 128);
            setLayer(3, OUTPUT_SIZE);
        } else {
            // Fallback for 3 layers
            setLayer(0, INPUT_SIZE);
            setLayer(1, 64);
            setLayer(2, OUTPUT_SIZE);
        }
        setWeights();
        System.out.println(" Engine Initialized: " + INPUT_SIZE + " In -> " + OUTPUT_SIZE + " Out");
    }




    //works imo
    public void setNeuralNetwork(int layers) {
        neuralNetwork = new NeuralNetwork(layers);
        if(layers>0) setLayer(0,INPUT_SIZE);
        if(layers>1) setLayer(layers-1,OUTPUT_SIZE);
        if(layers==STANDARD){
            setLayer(1,16);
            setLayer(2,16);

        }
        System.out.println("The neural network has been set");
        setWeights();
        System.out.println("The weights have been set");
        System.out.println("The neural network is ready");
        System.out.println("Number of layers: " +neuralNetwork.getNeurons().size());
    }


// SAME METHOD AS BELOW BUT WITH MORE COMMENTS USED DURING DEBUGGING. (I WILL MAKE A DEBUGGING FEATURE LATER ON)
//    public void setLayer(int layerIndex, int noOfNeurons) {
//        if(layerIndex>=neuralNetwork.getNeurons().size()) return;
//        //Clear existing neurons in the layer if any
//        neuralNetwork.getNeurons().get(layerIndex).clear();
//        for(int z=0; z<noOfNeurons;++z) {
//            Neuron neuron = new Neuron();
//            neuralNetwork.addNeuron(layerIndex,neuron);
//           // System.out.println("Adding neuron with bias: "+neuron.getBias() + " to layer "  + layerIndex);
//        }
//      //  System.out.println("number of neurons in layer " + layerIndex + " is " + noOfNeurons);
//    }

    public void setLayer(int layerIndex, int noOfNeurons) {
        if(layerIndex >= neuralNetwork.getNeurons().size()) return;

        neuralNetwork.getNeurons().get(layerIndex).clear();

        // Check if this is the last layer
        boolean isOutput = (layerIndex == neuralNetwork.getNeurons().size() - 1);

        for(int z=0; z<noOfNeurons; ++z) {
            Neuron neuron = new Neuron();
            neuron.setOutputNeuron(isOutput); // <--- Tell the neuron its job
            neuralNetwork.addNeuron(layerIndex, neuron);
        }
    }

    public void setWeights(){
        // initialize the weights
        neuralNetwork.setWeights(new Weights(new java.util.ArrayList<>()));


        int layers  = neuralNetwork.getNeurons().size();
        // we need weight between two layers (midLayer= 0 -> weights between layer 0 and layer 1)
        for(int midLayer = 0 ; midLayer<layers-1;++midLayer){

            neuralNetwork.getWeights().addMidLayer();   //adds a layer of weights in the weight list
            //neurons from one layer
            int neuronsInCurrentLayer = neuralNetwork.getNeurons().get(midLayer).size();

            //neurons from the next layer
            int neuronsInNextLayer = neuralNetwork.getNeurons().get(midLayer+1).size();

            //for every neuron in the current layer
            for(int neuronFrom=0;neuronFrom<neuronsInCurrentLayer;++neuronFrom){
                List<Float> connections = new ArrayList<>();

                double range = 1.0 / Math.sqrt(neuronsInCurrentLayer);
                //for every neuron in the next layer
                for(int neuronTo=0;neuronTo<neuronsInNextLayer;++neuronTo){
                    //randomize the weights between -range and range
                   // float weight = (float) (Math.random()*2*range-range);
                    float weight = (float) ((Math.random() * 2 * range - range) );
                    connections.add(weight);
                }
                neuralNetwork.getWeights().getWeightsOfLayer(midLayer).add(connections);
            }
        }

    }



    // forward propagation
    // Replace the forwardPass() method in NeuralEngine.java with this version

    public void forwardPass() {
        List<List<Neuron>> layers = neuralNetwork.getNeurons();
        Weights weights = neuralNetwork.getWeights();

        // Standard forward pass through hidden layers
        for(int prevLayer=0; prevLayer < layers.size()-1; ++prevLayer){
            List<Neuron> currentLayerNeurons = layers.get(prevLayer);
            List<Neuron> nextLayerNeurons = layers.get(prevLayer+1);

            for (int nextNeuron = 0; nextNeuron < nextLayerNeurons.size(); ++nextNeuron) {
                float sum = nextLayerNeurons.get(nextNeuron).getBias();

                for (int currentNeuron = 0; currentNeuron < currentLayerNeurons.size(); ++currentNeuron) {
                    float input = currentLayerNeurons.get(currentNeuron).getActivation();
                    float weight = weights.getWeight(prevLayer, currentNeuron, nextNeuron);
                    sum += weight * input;
                }
                nextLayerNeurons.get(nextNeuron).setVal(sum);
            }
        }

        // Apply Softmax to output layer for better probability distribution
        applySoftmax();
    }

    private void applySoftmax() {
        List<Neuron> outputLayer = neuralNetwork.getNeurons().getLast();

        // Find max for numerical stability
        float maxVal = Float.NEGATIVE_INFINITY;
        for(Neuron n : outputLayer) {
            if(n.getVal() > maxVal) maxVal = n.getVal();
        }

        // Compute exp and sum
        float sumExp = 0.0f;
        float[] expVals = new float[outputLayer.size()];
        for(int i = 0; i < outputLayer.size(); i++) {
            expVals[i] = (float)Math.exp(outputLayer.get(i).getVal() - maxVal);
            sumExp += expVals[i];
        }

        // Normalize to get probabilities
        for(int i = 0; i < outputLayer.size(); i++) {
            float probability = expVals[i] / sumExp;
            outputLayer.get(i).setActivation(probability);
        }
    }




    //turns out it is simpler to just do it in the training method itself here. I will switch this in python one though.
    //populate the inputs!
    public void setInput(float[] inputs){
        if(inputs.length!=INPUT_SIZE) return;
        for(int i=0;i<INPUT_SIZE;++i){
            neuralNetwork.getNeuron(0,i).setActivation(inputs[i]);
        }
        System.out.println("Inputs have been set");

    }

    public void train(int epochs) {
        // initialize the weights that are needed to be modified
        for (int e = 0; e < epochs; e++) {
            int iterations = trainingData.length / BATCH_SIZE;
            //no of the partitions you made of the data
            for (int i = 0; i < iterations; ++i) {


                float cost = 0;
                // the data in the given partition
                int start = i * BATCH_SIZE;
                //first populate the inputs
                for (int j = start; j < start + BATCH_SIZE; ++j) {
                    float[] inputs = trainingData[j];
                    List<Neuron> inputLayer = neuralNetwork.getNeurons().getFirst();
                    // k = 1 as the 0th index is the label
                    for (int k = 1; k < trainingData[j].length; ++k) {
                        //set the input layer of neurons
                        float input = inputs[k];
                        inputLayer.get(k - 1).setVal(input);
                    }

                    // do the forward pass once you have set the inputs
                    forwardPass();

                    cost += calculateCostFunction(j);

                    float label = inputs[0];
                    float[] targets = new float[OUTPUT_SIZE];
                    for (int x = 0; x < OUTPUT_SIZE; ++x) {
                        if (x == label) targets[x] = 1.0f;
                        else targets[x] = 0.f;
                    }

                    //We calculate the changes we want to make to each weight from the cost function we just made above.
                    backpropagate(targets);
                    if (j % 10 == 0) {
                        System.out.print("Target: " + label + " | Prediction: ");
                        List<Neuron> out = neuralNetwork.getNeurons().getLast();
                        int bestGuess = 0;
                        float bestVal = 0;
                        for(int k=0; k<10; k++) {
                            // Print the raw output to see if they are all 0.5 or 0.0
                            // System.out.print(String.format("%.2f ", out.get(k).getActivation()));
                            if(out.get(k).getActivation() > bestVal) {
                                bestVal = out.get(k).getActivation();
                                bestGuess = k;
                            }
                        }
                        System.out.println("-> " + bestGuess);
                    }
                }


                cost /= BATCH_SIZE;
                System.out.println("The cost is " + cost);
            }
            ++e;
        }

    }



    //I have overloaded the method for further customizability. You can specify your learning rate instead of using the default one

    private float learningRate;
    public void train(int epochs, float learningRate) {
        this.learningRate = learningRate;
        float bestCost = 1000.0f; // Track the best score
        int patience = 0;         // How long have we waited for improvement?

        for (int e = 0; e < epochs; e++) {

            // --- 1. Learning Rate Decay ---
            if (e > 0 && e % 500 == 0) { // Decay every 500 epochs
                this.learningRate *= 0.9f;
                System.out.println("   >>> Decay: Learning Rate is now " + this.learningRate);
            }

            int iterations = trainingData.length / BATCH_SIZE;
            float epochCost = 0;

            for (int i = 0; i < iterations; ++i) {
                int start = i * BATCH_SIZE;
                for (int j = start; j < start + BATCH_SIZE; ++j) {
                    float[] inputs = trainingData[j];
                    List<Neuron> inputLayer = neuralNetwork.getNeurons().getFirst();

                    // Set Inputs
                    for (int k = 1; k < trainingData[j].length; ++k) {
                        inputLayer.get(k - 1).setActivation(inputs[k]);
                    }
                    forwardPass();

                    // Accumulate cost for the epoch
                    epochCost += calculateCostFunction(j);

                    float label = inputs[0];
                    float[] targets = new float[OUTPUT_SIZE];
                    for (int x = 0; x < OUTPUT_SIZE; ++x) {
                        if (x == label) targets[x] = 1.0f;
                        else targets[x] = 0.f;
                    }

                    backpropagate(targets, this.learningRate);
                }


            }

            // Average cost for the whole epoch
            epochCost /= trainingData.length;

            // --- 2. SPEED UP: Only print every 100 epochs ---
            if (e % 100 == 0) {
                System.out.printf("Epoch %d | Cost: %.6f | LR: %.6f\n", e, epochCost, this.learningRate);
            }



            // --- 3. EARLY STOPPING ---
            // If cost drops significantly, reset patience
            if (epochCost < bestCost - 0.001f) {
                bestCost = epochCost;
                patience = 0;
            } else {
                patience++;
            }

            // If we haven't improved in 300 epochs, STOP.
            if (patience > 300) {
                System.out.println("\n------------------------------------------------");
                System.out.println("EARLY STOPPING TRIGGERED");
                System.out.println("Cost hasn't improved in 300 epochs. We are done.");
                System.out.println("Best Cost: " + bestCost);
                System.out.println("------------------------------------------------\n");
                break; // Exit the loop
            }


            // --- SHUFFLE DATA EVERY EPOCH (Prevents Memorization) ---
            for (int k = trainingData.length - 1; k > 0; k--) {
                int index = (int) (Math.random() * (k + 1));
                // Swap
                float[] temp = trainingData[index];
                trainingData[index] = trainingData[k];
                trainingData[k] = temp;
            }
        }
    }


    public int getPredictedDigit() {
        List<Neuron> outputLayer = neuralNetwork.getNeurons().getLast();
        int bestIndex = -1;
        float highestValue = -1f;

        for (int i = 0; i < outputLayer.size(); i++) {
            float activation = outputLayer.get(i).getActivation();
            if (activation > highestValue) {
                highestValue = activation;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    public void printConfidences() {
        // 1. Get the last layer (Output Layer)
        List<Neuron> outputLayer = neuralNetwork.getNeurons().getLast();

        System.out.print("CONFIDENCES:");

        // 2. Loop through all 10 neurons
        for (int i = 0; i < outputLayer.size(); i++) {
            // Print the activation (probability)
            System.out.print(outputLayer.get(i).getActivation());

            // Add a comma if it's not the last one
            if (i < outputLayer.size() - 1) {
                System.out.print(",");
            }
        }
        System.out.println(); // End the line
    }


    public void test(float[][] testData) {
        System.out.println("----------------------------------");
        System.out.println("Running Multi-Class Test (Geant4 Labels)...");

        int correctGuesses = 0;
        int[][] confusionMatrix = new int[OUTPUT_SIZE][OUTPUT_SIZE];

        // CORRECT GEANT4 ORDER: 0=Elec, 1=Pion, 2=Muon, 3=Gamma
        String[] classNames = {"Elec ", "Pion ", "Muon ", "Gamma"};

        for (int i = 0; i < testData.length; i++) {
            float[] row = testData[i];

            // 1. Set Inputs
            List<Neuron> inputLayer = neuralNetwork.getNeurons().getFirst();
            for(int k=1; k < row.length; ++k){
                inputLayer.get(k-1).setActivation(row[k]);
            }

            // 2. Forward Pass
            forwardPass();

            // 3. Get Prediction (ArgMax)
            List<Neuron> out = neuralNetwork.getNeurons().getLast();
            int predictedLabel = -1;
            float maxProb = -1.0f;

            for(int j=0; j<OUTPUT_SIZE; j++) {
                if(out.get(j).getActivation() > maxProb) {
                    maxProb = out.get(j).getActivation();
                    predictedLabel = j;
                }
            }

            // 4. Record
            int actualLabel = (int) row[0];
            if (actualLabel >= 0 && actualLabel < OUTPUT_SIZE) {
                confusionMatrix[actualLabel][predictedLabel]++;
                if (actualLabel == predictedLabel) correctGuesses++;
            }
        }

        // 5. Print Matrix
        System.out.println("\n--- CONFUSION MATRIX ---");
        System.out.print("Act \\ Pred | ");
        for(String name : classNames) System.out.print(name + " | ");
        System.out.println();

        for(int act=0; act<OUTPUT_SIZE; act++) {
            System.out.printf("%-10s | ", classNames[act]);
            for(int pred=0; pred<OUTPUT_SIZE; pred++) {
                System.out.printf("%-6d | ", confusionMatrix[act][pred]);
            }
            System.out.println();
        }

        double accuracy = (double) correctGuesses / testData.length * 100;
        System.out.println("\nFINAL ACCURACY: " + String.format("%.2f%%", accuracy));
        System.out.println("----------------------------------");
    }

    public void backpropagate(float[] targets){
            List<List<Neuron>> layers = neuralNetwork.getNeurons();
            Weights weights = neuralNetwork.getWeights();

            //We will store the errors of the current layer processed
            float[] errors = new float[OUTPUT_SIZE];

            //calculate the output layer error
            int outputLayerIndex = layers.size() - 1;
            // System.out.println(outputLayerIndex);
            List<Neuron> outputLayer = layers.get(outputLayerIndex);

            for (int x = 0; x < OUTPUT_SIZE; ++x) {
                Neuron neuron = outputLayer.get(x);
                float error = (neuron.getActivation() - targets[x]);
                errors[x] = error;

                //update the bias
                neuron.setBias(neuron.getBias() - LEARNING_RATE * error);
            }


            // calculate the errors in hidden layers
            for (int midLayer = layers.size() - 2; midLayer >= 0; --midLayer) {
                List<Neuron> currentLayerNeurons = layers.get(midLayer);
                List<Neuron> nextLayerNeurons = layers.get(midLayer + 1);

                // new array to store the errors of the current layer
                float[] currentLayerErrors = new float[currentLayerNeurons.size()];
                for (int currentNeuron = 0; currentNeuron < currentLayerNeurons.size(); ++currentNeuron) {
                    Neuron currentLayerNeuron = currentLayerNeurons.get(currentNeuron);

                    float errorSum = 0;

                    // calculate the gradient based on layer ahead
                    for (int nextNeuron = 0; nextNeuron < nextLayerNeurons.size(); ++nextNeuron) {
                        float weight = weights.getWeight(midLayer, currentNeuron, nextNeuron);

                        //the error sum that came from the coming layer
                        errorSum += weight * errors[nextNeuron];

                        //update the weight
                        float gradient = errors[nextNeuron] * currentLayerNeuron.getActivation();
                        float updatedWeight = weight - LEARNING_RATE * gradient;
                        weights.setWeight(midLayer, currentNeuron, nextNeuron, updatedWeight);

                    }

                    // calculate error for this neuron for layer before it
                    currentLayerErrors[currentNeuron] = errorSum * currentLayerNeuron.getSigmoidDerivative();

                    if (midLayer > 0) {
                        float newBias = currentLayerNeuron.getBias() - LEARNING_RATE * currentLayerErrors[currentNeuron];
                        currentLayerNeuron.setBias(newBias);
                    }


                }
                errors = currentLayerErrors;
            }


    }


// Add L2 regularization to prevent overfitting

    public void backpropagate(float[] targets, float learningRate){
        List<List<Neuron>> layers = neuralNetwork.getNeurons();
        Weights weights = neuralNetwork.getWeights();

        float L2_LAMBDA = 0.0001f; // Weight decay coefficient

        float[] errors = new float[OUTPUT_SIZE];

        // OUTPUT LAYER ERROR
        int outputLayerIndex = layers.size() - 1;
        List<Neuron> outputLayer = layers.get(outputLayerIndex);

        for (int x = 0; x < OUTPUT_SIZE; ++x) {
            Neuron neuron = outputLayer.get(x);
            float error = (neuron.getActivation() - targets[x]);
            errors[x] = error;
            neuron.setBias(neuron.getBias() - learningRate * error);
        }

        // HIDDEN LAYER BACKPROPAGATION WITH L2 REGULARIZATION
        for (int midLayer = layers.size() - 2; midLayer >= 0; --midLayer) {
            List<Neuron> currentLayerNeurons = layers.get(midLayer);
            List<Neuron> nextLayerNeurons = layers.get(midLayer + 1);

            float[] currentLayerErrors = new float[currentLayerNeurons.size()];

            for (int currentNeuron = 0; currentNeuron < currentLayerNeurons.size(); ++currentNeuron) {
                Neuron currentLayerNeuron = currentLayerNeurons.get(currentNeuron);
                float errorSum = 0;

                for (int nextNeuron = 0; nextNeuron < nextLayerNeurons.size(); ++nextNeuron) {
                    float weight = weights.getWeight(midLayer, currentNeuron, nextNeuron);
                    errorSum += weight * errors[nextNeuron];

                    // Calculate gradient
                    float gradient = errors[nextNeuron] * currentLayerNeuron.getActivation();
                    gradient = Math.max(-5.0f, Math.min(5.0f, gradient));

                    // L2 Regularization: Add weight decay term
                    // This penalizes large weights, preventing overfitting
                    float l2_penalty = L2_LAMBDA * weight;

                    float updatedWeight = weight - learningRate * (gradient + l2_penalty);
                    weights.setWeight(midLayer, currentNeuron, nextNeuron, updatedWeight);
                }

                currentLayerErrors[currentNeuron] = errorSum * currentLayerNeuron.getDerivative();

                if (midLayer > 0) {
                    float newBias = currentLayerNeuron.getBias() - learningRate * currentLayerErrors[currentNeuron];
                    currentLayerNeuron.setBias(newBias);
                }
            }

            errors = currentLayerErrors;
        }
    }




    private float calculateCostFunction(int trainingDataIndex){
        float[] trainingData = this.trainingData[trainingDataIndex];

        // Get the label
        int labelIndex = (int)trainingData[0];

        // Get output layer
        List<Neuron> outputLayer = neuralNetwork.getNeurons().getLast();

        // Cross-Entropy Loss: -log(predicted_probability_of_correct_class)
        float predictedProb = outputLayer.get(labelIndex).getActivation();

        // Clip to prevent log(0)
        predictedProb = Math.max(1e-7f, Math.min(1.0f - 1e-7f, predictedProb));

        // Negative log likelihood

        return -(float)Math.log(predictedProb);
    }




    public void setTrainingData(float[][] trainingData) {
        this.trainingData = trainingData;
    }





    // SAVING THE MODEL AFTER TRAINING

    public void saveModel(String filePath) throws FileNotFoundException {
        System.out.println("Saving model to " + filePath);
        try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(filePath))){

            // start storing number of layers
            List<List<Neuron>> layers = neuralNetwork.getNeurons();
            dos.writeInt(layers.size());

            for (List<Neuron> layer : layers) {
                dos.writeInt(layer.size());
                for (Neuron neuron : layer) {
                    //we only save the bias.
                    dos.writeFloat(neuron.getBias());
                }
            }

            Weights weights = neuralNetwork.getWeights();

            for (int x=0;x<layers.size()-1;++x){
                int currentLayerSize = layers.get(x).size();
                int nextLayerSize = layers.get(x+1).size();
                for (int from=0;from<currentLayerSize;++from){
                    for (int to=0;to<nextLayerSize;++to){
                        dos.writeFloat(weights.getWeight(x,from,to));
                    }
                }
            }

            System.out.println("Model saved!!");

        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error saving model");
        }
    }

    // LOADING THE MODEL AFTER TRAINING
    public static NeuralEngine loadModel(String filePath) throws IOException {
        System.out.println("Loading model from " + filePath);

        try(DataInputStream dis = new DataInputStream(new FileInputStream(filePath))){
            int numberOfLayers = dis.readInt();

            // 1. Create the engine
            NeuralEngine engine = new NeuralEngine(0);
            engine.neuralNetwork = new NeuralNetwork(numberOfLayers);

            // 2. Load Layers and Biases
            for (int x = 0; x < numberOfLayers; ++x){
                int numberOfNeurons = dis.readInt();
                engine.setLayer(x, numberOfNeurons);
                for (int y = 0; y < numberOfNeurons; ++y){
                    float bias = dis.readFloat();
                    engine.neuralNetwork.getNeuron(x, y).setBias(bias);
                }
            }

            // --- CRITICAL FIX: Update the Sizes! ---
            // We look at the loaded structure to determine the sizes
            engine.INPUT_SIZE = engine.neuralNetwork.getNeurons().getFirst().size();
            engine.OUTPUT_SIZE = engine.neuralNetwork.getNeurons().getLast().size();
            System.out.println("Model Geometry Loaded: " + engine.INPUT_SIZE + " -> " + engine.OUTPUT_SIZE);
            // ---------------------------------------

            // 3. Load Weights
            engine.setWeights();
            Weights weights = engine.neuralNetwork.getWeights();
            List<List<Neuron>> layers = engine.neuralNetwork.getNeurons();
            for (int x = 0; x < numberOfLayers - 1; ++x){
                int currentLayerSize = layers.get(x).size();
                int nextLayerSize = layers.get(x + 1).size();
                for (int from = 0; from < currentLayerSize; ++from){
                    for (int to = 0; to < nextLayerSize; ++to){
                        float weight = dis.readFloat();
                        weights.setWeight(x, from, to, weight);
                    }
                }
            }

            System.out.println("Model loaded!!");
            return engine;
        }
    }

}
