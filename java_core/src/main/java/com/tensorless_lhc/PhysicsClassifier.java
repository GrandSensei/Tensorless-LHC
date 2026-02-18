package com.tensorless_lhc;

import com.tensorless_lhc.tensorless.ExcelParse;
import com.tensorless_lhc.tensorless.NeuralEngine;
import com.tensorless_lhc.tensorless.NeuralNetwork;
import java.util.Random;

public class PhysicsClassifier {

    public static void main(String[] args) throws Exception {
        //test_and_train_std("data/training_data_Cu3.csv");
        test_model("src/main/resources/models/multiclass_classifier2.bin","data/training_data_Cu3.csv");
    }


    private static void test_model(String model_path,String test_data) throws Exception {
        // 1. LOAD DATA
        System.out.println("Loading Test Physics Data...");
        float[][] allData = ExcelParse.loadPhysicsData(test_data, -1);
        System.out.println("Loaded " + allData.length + " events.");

        // 2. SHUFFLE
        System.out.println("Shuffling data...");
        shuffleData(allData);

        NeuralEngine brain = NeuralEngine.loadModel(model_path);
        System.out.println("The input size is : "+brain.INPUT_SIZE);

        System.out.println("--- TEST SET (Unseen Data) ---");
        brain.test(allData);


    }

    private static void test_and_train_std(String filePath) throws Exception {
        // 1. LOAD DATA
        System.out.println("Loading Multi-Class Physics Data...");
        float[][] allData = ExcelParse.loadPhysicsData(filePath, -1);
        System.out.println("Loaded " + allData.length + " events.");

        // 2. SHUFFLE
        System.out.println("Shuffling data...");
        shuffleData(allData);

        // 3. TRAIN/TEST SPLIT (80/20)
        int splitIdx = (int)(allData.length * 0.8);
        float[][] trainData = new float[splitIdx][];
        float[][] testData = new float[allData.length - splitIdx][];

        System.arraycopy(allData, 0, trainData, 0, splitIdx);
        System.arraycopy(allData, splitIdx, testData, 0, testData.length);

        System.out.println("Training set: " + trainData.length + " samples");
        System.out.println("Test set: " + testData.length + " samples");

        // 4. INSPECT DATA
        System.out.println("\n--- TRAINING DATA INSPECTION ---");
        inspectData(trainData);
        System.out.println("------------------------------\n");

        // 5. CREATE NETWORK
        // 16 features -> 32 -> 16 -> 4 classes
        NeuralEngine brain = new NeuralEngine(16, 4, 4);

        brain.neuralNetwork = new NeuralNetwork(4);
        brain.setLayer(0, 16);   // Input: 10 layers + 6 engineered features
        brain.setLayer(1, 48);   // Hidden layer 1
        brain.setLayer(2, 32);   // Hidden layer 2
        brain.setLayer(3, 4);    // Output: 4 particle classes
        brain.setWeights();      // initialization

        brain.setTrainingData(trainData);
        brain.INPUT_SIZE=18;
        System.out.println("no of input size: "+brain.INPUT_SIZE);
        // 6. TRAIN with optimal learning rate
        System.out.println("Starting Training...");
        System.out.println("Architecture: 16 -> 48 -> 32 -> 4");
        System.out.println("Activation: ReLU (hidden) + Softmax (output)");
        System.out.println("Loss: Cross-Entropy with L2 Regularization");
        System.out.println("----------------------------------------\n");


        float time_start = System.currentTimeMillis();
        // Try different learning rates if needed:
        // 0.03 - Conservative (slow but stable)
        // 0.05 - Balanced (recommended)
        // 0.08 - Aggressive (fast but risky)
        brain.train(20000, 0.02f);

        // 7. TEST
        System.out.println("\n========================================");
        System.out.println("         FINAL EVALUATION");
        System.out.println("========================================\n");

        System.out.println("--- TEST SET (Unseen Data) ---");
        brain.test(testData);

        System.out.println("\n--- TRAINING SET (For Comparison) ---");
        brain.test(trainData);

        // 8. SAVE
        brain.saveModel("multiclass_classifier.bin");

        // 9. ANALYSIS
        System.out.println("\n========================================");
        System.out.println("         TRAINING COMPLETE");
        System.out.println("========================================");

        float end_time = System.currentTimeMillis();
        System.out.println("Training Time: " + (end_time-time_start)/1000.0);
        analyzeResults(trainData, testData, brain);
    }

    private static void inspectData(float[][] data) {
        int[] counts = new int[4];
        String[] names = {"Electron", "Pion", "Muon", "Gamma"};

        // Count each class
        for(float[] row : data) {
            int label = (int)row[0];
            if(label >= 0 && label < 4) counts[label]++;
        }

        System.out.println("CLASS DISTRIBUTION:");
        int total = data.length;
        for(int i=0; i<4; i++) {
            double pct = 100.0 * counts[i] / total;
            System.out.printf("  %s (label=%d): %d (%.1f%%)\n",
                    names[i], i, counts[i], pct);
        }

        // Check class balance
        int min = counts[0], max = counts[0];
        for(int c : counts) {
            if(c < min) min = c;
            if(c > max) max = c;
        }
        double imbalance = 1.0 - (double)min / max;
        System.out.printf("\nClass imbalance: %.1f%% ", imbalance * 100);
        if(imbalance < 0.1) {
            System.out.println("✓ Well balanced");
        } else {
            System.out.println("⚠ Consider balancing");
        }

        // Sample feature check
        System.out.println("\nSample features (first 3 of each class):");
        int[] shown = new int[4];
        for(float[] row : data) {
            int label = (int)row[0];
            if(shown[label] < 3) {
                System.out.printf("%s: L0=%.1f L1=%.1f L2=%.1f ... Ratio=%.3f EarlyFrac=%.3f, %.3f,%.3f\n",
                        names[label], row[1]*100, row[2]*100, row[3]*100,
                        row[12], row[14],row[15],row[16]);
                shown[label]++;
            }
            if(shown[0] >= 3 && shown[1] >= 3 && shown[2] >= 3 && shown[3] >= 3) break;
        }
    }

    private static void analyzeResults(float[][] trainData, float[][] testData, NeuralEngine brain) {
        double trainAcc = calculateAccuracy(trainData, brain);
        double testAcc = calculateAccuracy(testData, brain);
        double gap = trainAcc - testAcc;

        System.out.println("\nOVERFITTING ANALYSIS:");
        System.out.printf("  Train Accuracy: %.2f%%\n", trainAcc);
        System.out.printf("  Test Accuracy:  %.2f%%\n", testAcc);
        System.out.printf("  Gap:            %.2f%%\n", gap);

        if(gap < 3) {
            System.out.println("  Status: ✓ Excellent generalization");
        } else if(gap < 8) {
            System.out.println("  Status: ✓ Good generalization");
        } else if(gap < 15) {
            System.out.println("  Status: ⚠ Moderate overfitting");
        } else {
            System.out.println("  Status: ✗ Severe overfitting");
        }

        System.out.println("\nMULTI-ENERGY RECOMMENDATIONS:");

        if(testAcc < 70) {
            System.out.println("  • Accuracy is lower than expected for multi-energy:");
            System.out.println("    - Check if CSV contains incident energy column");
            System.out.println("    - Increase training epochs to 30000+");
            System.out.println("    - Check feature normalization (view first few rows)");
        } else if(testAcc < 78) {
            System.out.println("  • Reasonable performance. To improve:");
            System.out.println("    - Increase dataset size (360k+ events recommended)");
            System.out.println("    - Try learning rate 0.02 for faster convergence");
        } else if(testAcc < 85) {
            System.out.println("  ✓ Good performance for multi-energy classification!");
            System.out.println("  • To reach 85%+:");
            System.out.println("    - Ensure balanced energy distribution in dataset");
            System.out.println("    - Fine-tune layer sizes (try 19->80->40->4)");
        } else {
            System.out.println("  ✓ Excellent multi-energy performance!");
            System.out.println("  • Next steps:");
            System.out.println("    - Test on completely unseen energy (e.g., 750 MeV)");
            System.out.println("    - Generate confusion matrix per energy bin");
        }

        if(gap > 10) {
            System.out.println("\n  ⚠ OVERFITTING DETECTED:");
            System.out.println("    - Try reducing hidden layer sizes (64->48, 32->24)");
            System.out.println("    - Increase L2 regularization (lambda = 0.0005)");
            System.out.println("    - Add dropout (if implemented)");
            System.out.println("    - Get more diverse training data");
        }
    }




    private static double calculateAccuracy(float[][] data, NeuralEngine brain) {
        int correct = 0;
        for(float[] row : data) {
            // Set inputs
            for(int k = 1; k < row.length; k++) {
                brain.neuralNetwork.getNeuron(0, k-1).setActivation(row[k]);
            }
            brain.forwardPass();

            int predicted = brain.getPredictedDigit();
            int actual = (int)row[0];
            if(predicted == actual) correct++;
        }
        return 100.0 * correct / data.length;
    }

    private static void shuffleData(float[][] array) {
        Random rnd = new Random();
        for (int i = array.length - 1; i > 0; i--) {
            int index = rnd.nextInt(i + 1);
            float[] temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }
}






