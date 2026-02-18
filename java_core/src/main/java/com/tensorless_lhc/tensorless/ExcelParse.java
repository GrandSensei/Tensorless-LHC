package com.tensorless_lhc.tensorless;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class ExcelParse {

    //This code is refactored from a previous project of mine for some work stuff

    public static float[][] loadData(String path, int maxRows) throws Exception {
        List<float[]> data = new ArrayList<>();
        String line;

        try(BufferedReader br = new BufferedReader(new FileReader(path))){
            br.readLine();
            int count = 0;
            while((line = br.readLine()) != null){
                if(maxRows!=-1 && count>= maxRows) break;
                String[] values = line.split(",");

                float[] row = new float[values.length];
                row[0] = Float.parseFloat(values[0]);
                //convert the string to float
                for(int i=1;i<values.length;++i){
                    row[i] = Float.parseFloat(values[i])/255.f;
                }
                data.add(row);
                count++;

                if (count%1000 == 0) System.out.println("Loaded " + count + " rows");
            }

        }catch (Exception e){
            e.printStackTrace();
            throw new Exception("Error while reading the data file");
        }
        float[][] dataArray = new float[data.size()][data.getFirst().length];
        return data.toArray(dataArray);
    }




    public static float[][] loadPhysicsData(String path, int maxRows) throws Exception {
        System.out.println("Reading file with Enhanced Physics Features: " + path);
        List<float[]> rawDataList = new ArrayList<>();
        List<Float> rawLabels = new ArrayList<>();

        // For statistics (Pass 1)
        float sumRatio = 0, sumSqRatio = 0;
        float sumTotal = 0, sumSqTotal = 0;
        float sumPeakPos = 0, sumSqPeakPos = 0;
        int n = 0;

        float maxLogVal = 0.0f;

        // --- PASS 1: GATHER STATISTICS & LOG LAYERS ---
        try(BufferedReader br = new BufferedReader(new FileReader(path))){
            br.readLine();
            String line;
            int count = 0;
            while((line = br.readLine()) != null){
                if(maxRows != -1 && count >= maxRows) break;
                String[] values = line.split(",");

                // Label Mapping
                float rawLabel = Float.parseFloat(values[0]);
                if(rawLabel == 11) rawLabel = 0;       // e-
                else if(rawLabel == 211) rawLabel = 1; // pi-
                else if(rawLabel == 13) rawLabel = 2;  // mu-
                else if(rawLabel == 22) rawLabel = 3;  // gamma

                rawLabels.add(rawLabel);

                // Load raw layer values to calc stats
                float[] rawLayers = new float[10];
                float rowSum = 0;
                float maxEnergy = 0;
                int peakLayer = 0;

                for(int i=0; i<10; i++){
                    float val = Float.parseFloat(values[i+1]);
                    rawLayers[i] = val;
                    rowSum += val;

                    if(val > maxEnergy) {
                        maxEnergy = val;
                        peakLayer = i;
                    }
                }

                // Apply log scaling for stored layers
                float[] logLayers = new float[10];
                for(int i=0; i<10; i++){
                    float logVal = (float) Math.log(1.0 + rawLayers[i]);
                    logLayers[i] = logVal;
                    if(logVal > maxLogVal) maxLogVal = logVal;
                }

                // Calculate basic features for stats
                float ratio = (rowSum > 0) ? (rawLayers[0] / rowSum) : 0.0f;
                float peakPosition = peakLayer / 9.0f;

                // Accumulate
                sumTotal += rowSum;
                sumSqTotal += rowSum * rowSum;
                sumRatio += ratio;
                sumSqRatio += ratio * ratio;
                sumPeakPos += peakPosition;
                sumSqPeakPos += peakPosition * peakPosition;

                n++;
                rawDataList.add(logLayers);
                count++;
            }
        }

        // Calculate statistics
        float meanTotal = sumTotal / n;
        float stdTotal = (float) Math.sqrt((sumSqTotal / n) - (meanTotal * meanTotal));
        float meanRatio = sumRatio / n;
        float stdRatio = (float) Math.sqrt((sumSqRatio / n) - (meanRatio * meanRatio));
        float meanPeak = sumPeakPos / n;
        float stdPeak = (float) Math.sqrt((sumSqPeakPos / n) - (meanPeak * meanPeak));

        if(stdTotal == 0) stdTotal = 1;
        if(stdRatio == 0) stdRatio = 1;
        if(stdPeak == 0) stdPeak = 1;

        System.out.println("Feature Stats:");
        System.out.println("  Ratio - Mean: " + meanRatio + " Std: " + stdRatio);
        System.out.println("  Total - Mean: " + meanTotal + " Std: " + stdTotal);
        System.out.println("  Peak  - Mean: " + meanPeak + " Std: " + stdPeak);

        // --- PASS 2: BUILD FINAL DATASET ---
        // Feature Mapping:
        // [0]      : Label (0=e-, 1=pi-, 2=mu-, 3=gamma)
        // [1-10]   : Log-Scaled Layer Energies (Layers 0-9)
        // [11]     : Total Energy (Z-score)
        // [12]     : Layer 0 Ratio (Z-score)
        // [13]     : Peak Position (Z-score)
        // [14]     : Early Fraction (Raw)
        // [15]     : Layer 0 Interaction (Binary)
        // [16]     : Layer StdDev (Roughness)
        float[][] finalData = new float[rawDataList.size()][17];

        try(BufferedReader br = new BufferedReader(new FileReader(path))){
            br.readLine();
            int idx = 0;
            String line;
            while((line = br.readLine()) != null && idx < rawDataList.size()){
                String[] values = line.split(",");
                float[] logLayers = rawDataList.get(idx);

                // 1. RE-PARSE RAW DATA FIRST
                float[] rawLayers = new float[10];
                float rowSum = 0;
                float maxEnergy = 0;
                int peakLayer = 0;

                for(int i=0; i<10; i++){
                    rawLayers[i] = Float.parseFloat(values[i+1]);
                    rowSum += rawLayers[i];
                    if(rawLayers[i] > maxEnergy) {
                        maxEnergy = rawLayers[i];
                        peakLayer = i;
                    }
                }

                // 2. CALCULATE ADVANCED FEATURES (Now that rawLayers is full)

                // A. StdDev (Roughness) for Pion vs Muon
                float sum = 0;
                float sqSum = 0;
                for(float val : rawLayers) {
                    sum += val;
                    sqSum += val * val;
                }
                float mean = sum / 10.0f;
                float variance = (sqSum / 10.0f) - (mean * mean);
                if(variance < 0) variance = 0; // Floating point safety
                float layerStdDev = (float)Math.sqrt(variance);

                // B. Layer 0 Interaction for Gamma vs Electron
                // Gammas often have near 0 energy in layer 0. Electrons usually have signal.
                float layer0Interaction = (rawLayers[0] > 0.1f) ? 1.0f : 0.0f;

                // C. Standard Features
                float ratio = (rowSum > 0) ? (rawLayers[0] / rowSum) : 0.0f;
                float peakPosition = peakLayer / 9.0f;
                float earlySum = rawLayers[0] + rawLayers[1] + rawLayers[2];
                float earlyFraction = (rowSum > 0) ? (earlySum / rowSum) : 0.0f;

                // 3. FILL FINAL ARRAY
                finalData[idx][0] = rawLabels.get(idx);

                // Features 1-10: Log Layers
                for(int j=0; j<10; j++) {
                    finalData[idx][j+1] = logLayers[j] / maxLogVal;
                }

                // Features 11-16: Engineered
                finalData[idx][11] = (rowSum - meanTotal) / stdTotal;        // Total Energy (Z-score)
                finalData[idx][12] = (ratio - meanRatio) / stdRatio;         // Ratio (Z-score)
                finalData[idx][13] = (peakPosition - meanPeak) / stdPeak;    // Peak Pos (Z-score)
                finalData[idx][14] = earlyFraction;                          // Early Frac (0-1)
                finalData[idx][15] = layer0Interaction;                      // Interaction (Binary)
                finalData[idx][16] = layerStdDev / 100.0f;                   // Roughness (Scaled ~0-1)

                idx++;
            }
        }

        System.out.println("Loaded " + finalData.length + " events with 16 features (10 layers + 6 engineered)");
        return finalData;
    }




    //for checking
    public static String getFeatureName(int index) {
        if (index == 0) return "Label";
        if (index >= 1 && index <= 10) return "Log_Layer_" + (index - 1);

        return switch (index) {
            case 11 -> "Total_Energy_Z";
            case 12 -> "Layer0_Ratio_Z";
            case 13 -> "Peak_Pos_Z";
            case 14 -> "Early_Fraction";
            case 15 -> "Layer0_Interaction";
            case 16 -> "Roughness_StdDev";
            default -> "Unknown_Feature_" + index;
        };
    }

















}




