package com.namespace.services;


import com.namespace.tensorless.NeuralEngine; // Importing YOUR existing class
import com.namespace.tensorless.Neuron;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.List;


@Service
public class ParticleProcessor {

    @Autowired
    private DashboardService dashboard;


    private NeuralEngine brain; // Using YOUR NeuralEngine

    // --- NORMALIZATION CONSTANTS ---
    // These match the logic in ExcelParse.java but are hardcoded from your training stats.
    // This ensures the live data looks exactly like the training data.
    private static final float MEAN_TOTAL = 198.18878f;
    private static final float STD_TOTAL  = 65.41866f;

    private static final float MEAN_RATIO = 0.088917404f;
    private static final float STD_RATIO  = 0.1080856f;

    private static final float MEAN_PEAK  = 0.45594794f;
    private static final float STD_PEAK   = 0.28404868f;

    private static final float MAX_LOG_VAL = 6.0f;

    @PostConstruct
    public void init() {
        try {
            // 1. Get model from resources (Spring Boot way)
            // 1. Check for the Docker path first
            String modelPath = "";
            File dockerModel = new File("/app/models/multiclass_classifier2.bin");

            if (dockerModel.exists()) {
                System.out.println("🐳 Docker Detected: Loading model from " + dockerModel.getAbsolutePath());
                modelPath = dockerModel.getAbsolutePath();
            } else {
                // 2. Fallback to Local IDE path (Relative to project root)
                System.out.println("💻 IDE Detected: Searching local resources...");
                File localModel = new File("src/main/resources/models/multiclass_classifier2.bin");

                if (localModel.exists()) {
                    modelPath = localModel.getAbsolutePath();
                } else {
                    throw new RuntimeException("❌ Model file not found! Checked: \n" +
                            "1. " + dockerModel.getAbsolutePath() + "\n" +
                            "2. " + localModel.getAbsolutePath());
                }
            }

            System.out.println("✅ Loading Model from: " + modelPath);


            // 3. Load the brain using YOUR existing static method
            this.brain = NeuralEngine.loadModel(modelPath);
            System.out.println("🧠 AI BRAIN ONLINE: NeuralEngine Linked Successfully!");
            System.out.println("DEBUG: Brain expects " + brain.INPUT_SIZE + " inputs.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @KafkaListener(topics = "raw-particles", groupId = "particle-group")
    public void processEvent(String message) {
        if (brain == null) return;

        try {
            // Message format: "EventID,Label,E0,E1,E2,E3,E4,E5,E6,E7,E8,E9"
            String[] parts = message.split(",");

            // --- STEP 1: PARSE RAW DATA (Matching ExcelParse logic) ---
            float[] rawLayers = new float[10];
            float rowSum = 0;
            float maxEnergy = 0;
            int peakLayer = 0;

            for(int i=0; i<10; i++) {
                // Indices 2-11 are the energies
                rawLayers[i] = Float.parseFloat(parts[i+2]);
                rowSum += rawLayers[i];

                if(rawLayers[i] > maxEnergy) {
                    maxEnergy = rawLayers[i];
                    peakLayer = i;
                }
            }

            // --- STEP 2: FEATURE ENGINEERING (Replicating ExcelParse.java) ---
            // We must create the exact same float[16] array your model expects
            float[] inputs = new float[16];

            // Features 0-9: Log-Scaled Layers
            for(int i=0; i<10; i++) {
                inputs[i] = (float)Math.log(1.0 + rawLayers[i]) / MAX_LOG_VAL;
            }

            // Feature 10: Total Energy (Z-score)
            inputs[10] = (rowSum - MEAN_TOTAL) / STD_TOTAL;

            // Feature 11: Layer 0 Ratio (Z-score)
            float ratio = (rowSum > 0) ? (rawLayers[0] / rowSum) : 0.0f;
            inputs[11] = (ratio - MEAN_RATIO) / STD_RATIO;

            // Feature 12: Peak Position (Z-score)
            float peakPosition = peakLayer / 9.0f;
            inputs[12] = (peakPosition - MEAN_PEAK) / STD_PEAK;

            // Feature 13: Early Fraction (Raw)
            float earlySum = rawLayers[0] + rawLayers[1] + rawLayers[2];
            inputs[13] = (rowSum > 0) ? (earlySum / rowSum) : 0.0f;

            // Feature 14: Layer 0 Interaction (Binary)
            inputs[14] = (rawLayers[0] > 0.1f) ? 1.0f : 0.0f;

            // Feature 15: Roughness (Scaled)
            float mean = rowSum / 10.0f;
            float sqSum = 0;
            for(float val : rawLayers) sqSum += (val - mean) * (val - mean);
            float layerStdDev = (float)Math.sqrt(sqSum / 10.0f);
            inputs[15] = layerStdDev / 100.0f;

            // --- DEBUGGING INPUTS ---
            System.out.println("\n--- DEBUG: NEURAL NET INPUTS ---");
            System.out.print("[");
            for (float input : inputs) {
                System.out.printf(" %.2f", input);
            }
            System.out.println(" ]");
// ------------------------

            // --- STEP 3: INFERENCE (Using YOUR NeuralEngine) ---
            brain.setInput(inputs);
            brain.forwardPass();

            // Print the raw raw probabilities
            List<Neuron> outputs = brain.neuralNetwork.getNeurons().getLast();
            System.out.print("DEBUG RAW OUTPUTS: [ ");
            for(Neuron n : outputs) {
                System.out.printf("%.4f ", n.getActivation());
            }
            System.out.println("]");
            System.out.println("DEBUG CHECK -> Total Energy: " + rowSum + " MeV");
            int prediction = brain.getPredictedDigit();

            // --- STEP 4: REPORT ---
            String[] names = {"Electron", "Pion", "Muon", "Gamma"};
            String predictedName = names[prediction];
            String actualName = names[Integer.parseInt(parts[1])]; // The ground truth from Geant4
            String eventID = parts[0];

            // Accessing your neuralNetwork field to get confidence
            float confidence = brain.neuralNetwork.getNeurons().getLast().get(prediction).getActivation() * 100;


            // We construct a simple JSON string manually for speed
            String json = String.format(
                    "{\"id\":\"%s\", \"actual\":\"%s\", \"predicted\":\"%s\", \"conf\":%.1f, \"energy\":%.1f, \"correct\":%b}",
                    eventID, actualName, predictedName, confidence, rowSum, predictedName.equals(actualName)
            );

            dashboard.sendToDashboard(json);

            // Console Output with Color
            String color = (predictedName.equals(actualName)) ? "\u001B[32m" : "\u001B[31m";
            String reset = "\u001B[0m";

            System.out.println(color + "🔮 PREDICTION: " + predictedName +
                    " \t(Actual: " + actualName + ")" +
                    " \t| Conf: " + String.format("%.1f", confidence) + "%" +
                    " \t| ID: " + eventID + reset);

        } catch (Exception e) {
            System.err.println("Error processing event: " + e.getMessage());
        }
    }
}