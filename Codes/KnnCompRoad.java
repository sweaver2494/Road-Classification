/*
 * This code makes a comparison of nearest neighbors (Euclidean distance)
 * for randomly selected test instances calculated before and after the 
 * application of PCA pre-processing.
 *
 * This code works for road classification datasets
 *
 * @author Scott Weaver
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import javafx.util.Pair;

public class KnnCompRoad {
    private static final int NUM_OF_NEIGHBORS = 5;
    
    private static String FEATUREFILEPATH = "Data/FeatureFiles/features_0_nomag.csv";
    //Must include. This is the file you want to test against feature files
    private static String TESTFILEPATH = "Data/TestData/testData.csv";
    
    public static void main(String[] args) {
		System.out.println("Feature File Path " + FEATUREFILEPATH);
		System.out.println("Test File Path " + TESTFILEPATH);
		
		if ((new File(TESTFILEPATH)).isFile() && (new File(FEATUREFILEPATH)).isFile()) {
			performClassificationAnalysis(FEATUREFILEPATH, TESTFILEPATH);
		} else {
			System.err.println("Test File or Feature File does not exist");
		}
    }
    
    private static ArrayList<String> getColumnHeaders(String line) {
    	ArrayList<String> columnHeaders = new ArrayList<>();
    	
    	String dataCompsStr[] = line.substring(line.indexOf(",") + 1).split(",");
    	
    	for (String feature : dataCompsStr) {
    		columnHeaders.add(feature);
    	}
    	
    	return columnHeaders;
    }
    
    public static void performClassificationAnalysis(String featureFilePath, String testFilePath) {
        try {
        	//Contains each row of variables
            ArrayList<double[]> fullData = new ArrayList<>();
            ArrayList<String> fullDataClassification = new ArrayList<>();
	        BufferedReader brFeature = new BufferedReader(new FileReader(featureFilePath));
	        BufferedReader brTest = new BufferedReader(new FileReader(testFilePath));
	        
	        String line = brFeature.readLine();
	        brTest.readLine();
	        
	        ArrayList<String> columnHeaders = getColumnHeaders(line);

	        line = brFeature.readLine();
	        int dataSize = line.length() - line.replace(",", "").length();
	
	        double dataAvg[] = new double[dataSize];
	
	        while (line != null) {
	        	String classification = line.substring(0, line.indexOf(","));
	            String dataCompsStr[] = line.substring(line.indexOf(",") + 1).split(",");
	
	            double dataComps[] = new double[dataSize];
	
	            for (int i = 0; i < dataSize; i++) {
	                dataComps[i] = Double.parseDouble(dataCompsStr[i]);
	                dataAvg[i] += dataComps[i];
	            }
	
	            fullData.add(dataComps);
	            fullDataClassification.add(classification);
	            line = brFeature.readLine();
	        }
	        brFeature.close();
	        
	        //One line of data on test file
	        line = brTest.readLine();
	        String testDataClassification = line.substring(0, line.indexOf(","));
	        String testDataStr[] = line.substring(line.indexOf(",") + 1).split(",");
	        double testData[] = new double[dataSize];
	        for (int i = 0; i < dataSize; i++) {
                testData[i] = Double.parseDouble(testDataStr[i]);
            }
            brTest.close();
	        

	        testPCA(fullData, testData, fullDataClassification, testDataClassification, dataAvg, columnHeaders);
	        
        } catch(IOException e) {
        	System.err.println("Cannot read feature file.");
        }
    }
    
    public static void testPCA(ArrayList<double[]> fullData, double[] testData, ArrayList<String> fullDataClassification, String testDataClassification, double[] dataAvg, ArrayList<String> columnHeaders) {
    	int dataSize = testData.length;
    	
    	//Perform KNN classification without PCA on original data
    	System.out.println("---------------------------------------------");
    	System.out.println("Number of Components = " + Integer.toString(dataSize));
    	
    	printNearestNeighbors(fullData, testData, fullDataClassification, testDataClassification, dataAvg, false, columnHeaders, dataSize);
    	System.out.println("---------------------------------------------");
    	
    	//Perform KNN classification after PCA on original data. For N dimensions, repeat N times, each time decrementing N.
    	for (int reducedSize = dataSize; reducedSize > 0; reducedSize--) {
	    	System.out.println("Number of Components = " + Integer.toString(reducedSize));
	        
	    	//the false-true boolean specifies whether or not to perform PCA
	        printNearestNeighbors(fullData, testData, fullDataClassification, testDataClassification, dataAvg, true, columnHeaders, reducedSize);
	            
	        System.out.println("---------------------------------------------");
    	}
    }

    public static double calculateDistance(double[] array1, double[] array2) {
        double Sum = 0.0;
        for (int i = 0; i < array1.length; i++) {
            Sum = Sum + Math.pow((array1[i] - array2[i]), 2.0);
        }
        return Math.sqrt(Sum);
    }

    private static void printNearestNeighbors(ArrayList<double[]> fullData, double[] testData, ArrayList<String> fullDataClassification, String testDataClassification, double[] dataAvg, boolean performPCA, ArrayList<String> columnHeaders, int reducedLength) {
    	
    	ArrayList<DistObj> distObjects = null;
    	ArrayList<Pair<Integer, Double>> covarianceList = new ArrayList<>();
    	
        if (performPCA) {
            int fullDataSize = fullData.size();
            int dataSize = fullData.get(0).length;
            
	        for (int i = 0; i < dataSize; i++) {
	            dataAvg[i] /= fullDataSize;
	        }
	        
	        Pair<ArrayList<double[]>, double[]> newData = MLUtilities.performPCA(fullData, testData, dataAvg, reducedLength, covarianceList);
	        
	    	System.out.println();
	        for (Pair<Integer, Double> item : covarianceList) {
	        	System.out.print(columnHeaders.get(item.getKey()) + "(" + item.getValue() + "), ");
	        }
	        System.out.println();
	        
	        System.out.println("\nNeighbors After PCA: \n");
	        
	        distObjects = MLUtilities.performKNN(newData.getKey(), newData.getValue());
        } else {
        	System.out.println("\nNeighbors Before PCA: \n");

        	distObjects = MLUtilities.performKNN(fullData, testData);
        }
        
        System.out.println("Test Data Classification: " + testDataClassification);
        double matchingNeighbors = 0;
        double totalDistance = 0;

        for (int i = 0; i < NUM_OF_NEIGHBORS; i++) {
        	int index = distObjects.get(i).index;
        	double distance = distObjects.get(i).distance;
            System.out.println("Neighbor " + (i+1) + ": Index=" + index + ", Classification=" + fullDataClassification.get(index) + ", Distance=" + distance);
            
            totalDistance += distance;
            if (fullDataClassification.get(index).equals(testDataClassification)) {
            	matchingNeighbors++;
            }
        }
        
        double classificationAccuracy = (matchingNeighbors / NUM_OF_NEIGHBORS) * 100;
        System.out.println("Number of nearest neighbors correctly classified: " + matchingNeighbors);
        System.out.println("Classification Accuracy: " + classificationAccuracy + "%");
        System.out.println("Sum of distances:" + totalDistance);
        
    }
    
}