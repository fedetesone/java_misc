package com.annoypoc;

import com.spotify.annoy.ANNIndex;
import com.spotify.annoy.IndexType;
import org.apache.commons.math3.random.JDKRandomGenerator;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class AnnoyPoc {

    private static final int NUMBER_DIMENSIONS = 200;
    private static final IndexType indexType = IndexType.ANGULAR;

    private static final int NUMBER_RESULTS = 10;

    public void test() throws IOException {
        final ANNIndex index = loadIndex();

        final GaussianRandomNumberGenerator generator = new GaussianRandomNumberGenerator(NUMBER_DIMENSIONS, new JDKRandomGenerator());
        float[] queryVector = generator.nextVector();
        System.out.println("Random vector: " + Arrays.toString(queryVector));

        getNearest(queryVector, index);
        getPqEntries(queryVector, index);

        // Default example
        final String filename = Objects.requireNonNull(this.getClass().getClassLoader().getResource("annoy_index.ann")).getPath();

        String[] args = {filename, String.valueOf(NUMBER_DIMENSIONS), "angular", String.valueOf(20)};
        ANNIndex.main(args);
    }

    private void getNearest(float[] queryVector, ANNIndex index) {
        List<Integer> nearestVectors = index.getNearest(queryVector, NUMBER_RESULTS);
        nearestVectors.forEach(System.out::println);
    }

    private void getPqEntries(float[] queryVector, ANNIndex index) {
        List<ANNIndex.PQEntry> entries = index.getNearestPqEntries(queryVector, NUMBER_RESULTS);
        entries.forEach(System.out::println);
    }

    private ANNIndex loadIndex() throws IOException {
        final String filename = Objects.requireNonNull(this.getClass().getClassLoader().getResource("annoy_index.ann")).getPath();
        return new ANNIndex(NUMBER_DIMENSIONS, filename, indexType);
    }

}
