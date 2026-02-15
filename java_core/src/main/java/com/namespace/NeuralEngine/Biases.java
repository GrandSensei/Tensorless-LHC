package com.namespace.NeuralEngine;

public class Biases {

    private float[] biases;

    public Biases(int size) {
        biases = new float[size];
    }

    public float[] getBiases() {
        return biases;
    }

    public void setBiases(float[] biases) {
        this.biases = biases;
    }

    //there is a better function I will think of in a while

    public void randomizeBiases() {
        for (int i = 0; i < biases.length; i++) {
            biases[i] = (float) Math.random() * 2 - 1;
        }
    }

}
