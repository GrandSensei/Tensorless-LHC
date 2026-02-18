package com.tensorless_lhc.tensorless;

public class Neuron {

    private float val;
    private boolean isOutputNeuron = false;
    private float activation;
    private float bias;

    public Neuron(float val) {
        this.val = val;
    }

    public float getVal() {
        return val;
    }

    public float getBias() {
        return bias;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }

    public void setVal(float val) {
        this.val = val;
        activate(val);
    }

    public float getActivation() {
        return activation;
    }

    public void setActivation(float activatedValue) {
        this.activation = activatedValue;
    }

    public Neuron() {
        setVal(0);
        setRandomBias();
    }

    public void setRandomBias() {
        // Small random bias instead of constant 0.01
        this.bias = (float)(Math.random() * 0.2 - 0.1); // Range: -0.1 to 0.1
    }

    public void activate(float val){
        this.val = val;
        if (isOutputNeuron) {
            // Softmax-friendly: Keep raw values, apply softmax in forward pass
            // For now, use sigmoid but with better scaling
            this.activation = (float)(1.0 / (1.0 + Math.exp(-val)));
        } else {
            // Standard ReLU (not leaky) - cleaner gradients
            this.activation = Math.max(0.0f, val);
        }
    }

    public float getSigmoidDerivative(){
        return (this.activation * (1 - this.activation));
    }

    public float getDerivative(){
        if (isOutputNeuron) {
            // Sigmoid Derivative
            return (this.activation * (1 - this.activation));
        } else {
            // ReLU Derivative: 1 if active, 0 if dead
            return (val > 0) ? 1.0f : 0.0f;
        }
    }

    public void setOutputNeuron(boolean isOutput) {
        this.isOutputNeuron = isOutput;
    }
}