/*
 * This code makes a comparision of nearest neighbours (euclidean distance)
 * for randomly selected testinstances calculated before and after the 
 * application of PCA pre-processing.
 *
 * This code works for road classification datasets
 *
 * @author Scott Weaver
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class KnnCompRoad {
    private static final int NUM_OF_NEIGHBOURS = 4;
    //private static int TEST_INDEX = 2;
    private static int REDUCED_LENGTH = 8;
    
    //Enter path of feature file if using existing file. Otherwise, a new one is created using files in RawData folder.
    private static String FEATUREFILEPATH = "Data/FeatureFiles/features_0.csv";
    
    /*******Must include. This is the file you want to test against feature files*******/
    private static String TESTFILEPATH = "Data/TestData/testData.csv";

    public static void main(String[] args) {
    	String featureFolderPath = "Data/FeatureFiles/";
    	(new File(featureFolderPath)).mkdirs();

    	//If using an existing feature file, do not create new one. Otherwise file is generated from rawData files.
    	if (FEATUREFILEPATH.equals("") || !(new File(FEATUREFILEPATH)).isFile()) {
			//Ensure files have unique names by appending an integer after conflicting filenames.
			int i=0;
			String featureFilePath  = featureFolderPath + "features_0.csv";
			while ((new File(featureFilePath)).exists()) {
				i++;
				featureFilePath  = featureFolderPath + "features_" + i + ".csv";
			}
			FEATUREFILEPATH = featureFilePath;

	    	//values contains the values of each sensor type
	        HashMap<String, ArrayList<Double>> values = new HashMap<String, ArrayList<Double>>();
	        //rms is the root-means-squared of all values of each sensor type
	        HashMap<String, Double> rms = new HashMap<String, Double>();
	        //avg is the average of all values of each sensor type
	        HashMap<String, Double> avg = new HashMap<String, Double>();
	        //sdv is the standard deviation of all values of each sensor type
	        HashMap<String, Double> sdv = new HashMap<String, Double>();
	        
	        
	    	String rawDataFolderPath = "Data/RawData/";
	    	File rawDataFolder = new File(rawDataFolderPath);
	    	
	    	//Calculate sensor average & variance for each feature, then write to features file
			try {
		    	BufferedWriter featureFileBuffer = new BufferedWriter(new FileWriter(FEATUREFILEPATH, true));

		    	int row = 0;
		    	if (rawDataFolder.isDirectory()) {
		    		for (File rawDataFile : rawDataFolder.listFiles()) {
		    			
		    			//read sensor data file and list all values for each feature
		    	        String classification = readRawData(rawDataFile, values);
		    	        
		    	        //calculate the average, root-means-squared, and standard deviation of all values for each feature
		    	        calculateStatistics(values, avg, rms, sdv);
		    	        
		    	        //write feature column headers only once
		    	        if (row == 0) {
		    	        	writeFeatureFileHeaders(featureFileBuffer, avg, rms, sdv);
		    	        }
		    	        
		    	        //write single row of feature data, one column per feature
		    	        writeFeatureFile(featureFileBuffer, avg, rms, sdv, classification);
		    	        
		    	        //set all default values for each key to reuse containers with the same key ordering
		    	        for (String key : values.keySet()) {
		    	        	ArrayList<Double> tempArrayList = values.get(key);
		    	        	tempArrayList.clear();
		    	        	
		    	        	avg.replace(key, (double)0);
		    	        	rms.replace(key, (double)0);
		    	        	sdv.replace(key, (double)0);
		    	        }
		    	        
		    	        row++;
		    		}
		    	}
		    	featureFileBuffer.close();
		    	
		    	
		    	
			} catch (IOException e) {
				System.err.println("IO Exception: " + e.getMessage());
			}
    	}
		
		System.out.println("Feature File Path " + FEATUREFILEPATH);
		System.out.println("Test File Path " + TESTFILEPATH);
		
		if ((new File(TESTFILEPATH)).isFile()) {
			performClassificationAnalysis(FEATUREFILEPATH, TESTFILEPATH);
		} else {
			System.err.println("Test File does not exist");
		}
    }
    
    //Read rawData files and record all values for each feature type
    public static String readRawData(File rawDataFile, HashMap<String, ArrayList<Double>> values) {
    	String classification = "";
    	try {
	        BufferedReader rawDataBuffer = new BufferedReader(new FileReader(rawDataFile));
	        String line = rawDataBuffer.readLine();
	        //read classification from first row
	        if (line != null) {
	        	classification = line;
	        	line = rawDataBuffer.readLine();
	        }
	        //read all data from file
	        while (line != null) {
	        	String dataCompsStr[] = line.split(",");
	        	String key = dataCompsStr[0];
	        	double val = Double.parseDouble(dataCompsStr[2]);
	        	
	        	if (values.containsKey(key)) {
	        		ArrayList<Double> temp = values.get(key);
	        		temp.add(val);
	        	} else {
	        		ArrayList<Double> temp = new ArrayList<Double>();
	        		temp.add(val);
	        		values.put(key, temp);
	        	}
	        	
	        	line = rawDataBuffer.readLine();
	        }
	        rawDataBuffer.close();
		} catch (IOException e) {
			System.err.println("Cannot read raw data file: " + rawDataFile.getAbsolutePath());
		}
    	return classification;
    }
    
    
    public static void calculateStatistics(HashMap<String, ArrayList<Double>> values, HashMap<String, Double> avg, HashMap<String, Double> rms, HashMap<String, Double> sdv) {
        for (String key : values.keySet()) {
        	//calculate average for each reading (value) for each sensor type (key)
        	for (double val : values.get(key)) {
        		if (avg.containsKey(key)) {
                	avg.put(key, avg.get(key) + val);
                } else {
                	avg.put(key, val);
                }
        	}
        	double keyAvg = avg.get(key) / values.get(key).size();
        	avg.put(key, keyAvg);
        	
        	//calculate root-means-squared for each reading (value) for each sensor type (key)
        	//root-means-squared is square root of the average of squared values
        	for (double val : values.get(key)) {
        		if (rms.containsKey(key)) {
        			rms.put(key, rms.get(key) + Math.pow(val, 2));
        		} else {
                   	rms.put(key, Math.pow(keyAvg, 2));
                }
        	}
        	double keyRms = rms.get(key) / values.get(key).size();
        	keyRms = Math.sqrt(keyRms);
        	rms.put(key, keyRms);
        	
        	//calculate standard deviation for each reading (value) for each sensor type (key)
        	//standard deviation is the square root of the average of the squared differences from the mean
        	for (double val : values.get(key)) {
        		if (sdv.containsKey(key)) {
        			sdv.put(key, sdv.get(key) + Math.pow(val - keyAvg, 2));
        		} else {
                   	sdv.put(key, Math.pow(val - keyAvg, 2));
                }
        	}
        	double keySdv = sdv.get(key) / values.get(key).size();
        	keySdv = Math.sqrt(keySdv);
        	sdv.put(key, keySdv);
        }
    }
     
    //assumes values exist for sum, avg, and var
    public static void writeFeatureFileHeaders(BufferedWriter featureFileBuffer, HashMap<String, Double> avg, HashMap<String, Double> rms, HashMap<String, Double> sdv) {
		try {
			featureFileBuffer.write("classification,");
			
	    	//Write column headers.
			for (String key : avg.keySet()) {
				featureFileBuffer.write(key + "_avg,");
			}
			
			for (String key : rms.keySet()) {
				featureFileBuffer.write(key + "_rms,");
			}
			
			int i = 0;
			for (String key : sdv.keySet()) {
				i++;
				featureFileBuffer.write(key + "_sdv");
				
				//don't write comma after last feature
				if (i < sdv.keySet().size()) {
					featureFileBuffer.write(",");
				}
			}
			featureFileBuffer.newLine();
			featureFileBuffer.flush();
		} catch (IOException e) {
			System.err.println("Cannot write feature file headers.");
		}
    }
    
    public static void writeFeatureFile(BufferedWriter featureFileBuffer, HashMap<String, Double> avg, HashMap<String, Double> rms, HashMap<String, Double> sdv, String classification) {
    	try {
    		featureFileBuffer.write(classification + ",");
    		
	    	//Write all values for single row.
			for (String key : avg.keySet()) {
				double val = avg.get(key);
				featureFileBuffer.write(Double.toString(val) + ",");
				System.out.println("Avg: " + Double.toString(val));
			}
			for (String key : rms.keySet()) {
				double val = rms.get(key);
				featureFileBuffer.write(Double.toString(val) + ",");
				System.out.println("Rms: " + Double.toString(val));
			}
			int i = 0;
			for (String key : sdv.keySet()) {
				i++;
				double val = sdv.get(key);
				featureFileBuffer.write(Double.toString(val));
				System.out.println("Sdv: " + Double.toString(val));
				
				//don't write comma after last feature
				if (i < sdv.keySet().size()) {
					featureFileBuffer.write(",");
				}
			}
			featureFileBuffer.newLine();
			featureFileBuffer.flush();
    	} catch (IOException e) {
    		System.err.println("Cannot write to feature file.");
    	}
    }
    
    public static void performClassificationAnalysis(String featureFilePath, String testFilePath) {
        try {
        	//Contains each row of variables
            ArrayList<double[]> fullData = new ArrayList<>();
            ArrayList<String> fullDataClassification = new ArrayList<>();
	        BufferedReader brFeature = new BufferedReader(new FileReader(featureFilePath));
	        BufferedReader brTest = new BufferedReader(new FileReader(testFilePath));
	        
	        //Ignore headers on first row. Skip to data
	        brFeature.readLine();
	        brTest.readLine();

	        String line = brFeature.readLine();
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
	        String classification = line.substring(0, line.indexOf(","));
	        String testCompsStr[] = line.substring(line.indexOf(",") + 1).split(",");
	        double testComps[] = new double[dataSize];
	        for (int i = 0; i < dataSize; i++) {
                testComps[i] = Double.parseDouble(testCompsStr[i]);
                dataAvg[i] += testComps[i];
            }
	        fullData.add(testComps);
            fullDataClassification.add(classification);
            brTest.close();
	        

	        testPCA(fullData, fullDataClassification, dataAvg, dataSize);
	        
        } catch(IOException e) {
        	System.err.println("Cannot read feature file.");
        }
    }
    
    public static void testPCA(ArrayList<double[]> fullData, ArrayList<String> fullDataClassification, double[] dataAvg, int dataSize) {
    	for (int i = dataSize; i > 0; i--) {
	    	System.out.println("---------------------------------------------");
	    	System.out.println("Number of Components = " + Integer.toString(i));
	    	
	    	REDUCED_LENGTH = i;
	        
	    	//the false-true boolean specifies whether or not to perform PCA
	    	printNearestNeighbors(fullData, fullDataClassification, dataAvg, false);
	        printNearestNeighbors(fullData, fullDataClassification, dataAvg, true);
	            
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

    private static void printNearestNeighbors(ArrayList<double[]> fullData, ArrayList<String> fullDataClassification, double[] dataAvg, boolean performPCA) {

        if (performPCA) {
            int fullDataSize = fullData.size();
            int dataSize = fullData.get(0).length;
            
	        for (int i = 0; i < dataSize; i++) {
	            dataAvg[i] /= fullDataSize;
	        }
	        fullData = MLUtilities.performPCA(fullData, dataAvg, REDUCED_LENGTH);
	        
	        System.out.println("\nNeighbors After PCA: \n");
        } else {
        	System.out.println("\nNeighbors Before PCA: \n");
        }

        ArrayList<DistObj> distObjects = MLUtilities.performKNN(fullData);

        for (int i = 1; i <= NUM_OF_NEIGHBOURS; i++) {
        	int index = distObjects.get(i).index;
        	double distance = distObjects.get(i).distance;
            System.out.println("Neighbor " + i + ": Index=" + index + ", Classification=" + fullDataClassification.get(index) + ", Distance=" + distance);
        }
    }
    
}