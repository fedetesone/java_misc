package com.annoypoc;

import org.apache.commons.math3.random.RandomGenerator;

public class GaussianRandomNumberGenerator {

    private final int N;
    private final RandomGenerator rg;

    public GaussianRandomNumberGenerator(int N, RandomGenerator rg) {
        this.N = N;
        this.rg = rg;
    }

    public float nextNormalizedFloat() {
        return rg.nextFloat();
    }

    public float[] nextVector() {
        float[] vector = new float[N];
        for (int i = 0; i < vector.length; ++i) {
            vector[i] = nextNormalizedFloat();
        }
        return vector;
    }
}
