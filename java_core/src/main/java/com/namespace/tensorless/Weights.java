package com.namespace.tensorless;

import java.util.ArrayList;
import java.util.List;

public class Weights {

    //consider a 3D list for now
    //some random words


    private List<List<List<Float>>> weights;

   public Weights(List<List<List<Float>>> weights) {
    this.weights = weights;
   }

   public float getWeight(int x, int y, int z) {
       return weights.get(x).get(y).get(z);
   }


   public void setWeight(int x, int y, int z, float weight) {
      weights.get(x).get(y).set(z, weight);
   }

   public void changeWeight(int layer, int neuron1, int neuron2 , float val){
       float w = weights.get(layer).get(neuron1).get(neuron2) -val;
       weights.get(layer).get(neuron1).set(neuron2,w);

   }

   public void addWeight(int layer, int neuron1,float val){
       weights.get(layer).get(neuron1).add(val);
   }

   public void addMidLayer(){
       weights.add(new ArrayList<>());
   }

   public List<Float> getWeightsOfNeuron(int layer, int neuron1) {
       return weights.get(layer).get(neuron1);
   }

   public List<List<Float>> getWeightsOfLayer(int layer) {
       return weights.get(layer);
   }



}
