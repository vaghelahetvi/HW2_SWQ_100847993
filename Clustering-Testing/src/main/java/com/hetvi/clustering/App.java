package com.hetvi.clustering;

import java.io.File;
import net.sf.javaml.clustering.*;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;
import net.sf.javaml.clustering.evaluation.*;

public class App {
    public static void main(String[] args) throws Exception {
        // Load the Iris dataset
        Dataset data = FileHandler.loadDataset(new File("datasets/iris.data"), 4, ",");

        // Initialize clustering algorithms
        Clusterer kMeans = new KMeans();
        Clusterer aqbc = new AQBC();
        Clusterer som = new SOM();

        // Perform clustering with KMeans
        long startTimeKMeans = System.nanoTime();
        Dataset[] clustersKMeans = kMeans.cluster(data);
        long durationKMeans = System.nanoTime() - startTimeKMeans;

        // Perform clustering with AQBC
        long startTimeAQBC = System.nanoTime();
        Dataset[] clustersAQBC = aqbc.cluster(data);
        long durationAQBC = System.nanoTime() - startTimeAQBC;

        // Perform clustering with SOM
        long startTimeSOM = System.nanoTime();
        Dataset[] clustersSOM = som.cluster(data);
        long durationSOM = System.nanoTime() - startTimeSOM;

        // Evaluate clusters using different metrics
        ClusterEvaluation aic = new AICScore();
        ClusterEvaluation bic = new BICScore();
        ClusterEvaluation sse = new SumOfSquaredErrors();
        ClusterEvaluation saps = new SumOfAveragePairwiseSimilarities();

        // Print cluster evaluation scores for each clustering method
        printEvaluationScores("KMeans", clustersKMeans, aic, bic, sse, saps);
        printEvaluationScores("AQBC", clustersAQBC, aic, bic, sse, saps);
        printEvaluationScores("SOM", clustersSOM, aic, bic, sse, saps);

        // Print execution time for each clustering method
        System.out.println("Execution time (ms): KMeans: " + durationKMeans / 1e6 + ", AQBC: " + durationAQBC / 1e6
                + ", SOM: " + durationSOM / 1e6);
    }

    private static void printEvaluationScores(String algorithmName, Dataset[] clusters, ClusterEvaluation aic,
            ClusterEvaluation bic, ClusterEvaluation sse, ClusterEvaluation saps) {
        System.out.println("\n" + algorithmName + " Clustering Evaluation:");
        System.out.println("AIC Score: " + aic.score(clusters));
        System.out.println("BIC Score: " + bic.score(clusters));
        System.out.println("Sum of Squared Errors: " + sse.score(clusters));
        System.out.println("Sum of Average Pairwise Similarities: " + saps.score(clusters));
    }
}
