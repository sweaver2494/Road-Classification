/*
 * This code makes a comparision of nearest neighbours (euclidean distance)
 * for randomly selected testinstances calculated before and after the 
 * application of PCA pre-processing.
 *
 * This code works for road classification datasets
 *
 * @author Scott Weaver
 */

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
//import java.util.Random;


public class KnnCompRoad {
    private static final int NUM_OF_NEIGHBOURS = 4;
    //private static int TEST_INDEX = 2;
    private static int REDUCED_LENGTH = 8;
    
    //Enter path of feature file if using existing file. Otherwise, a new one is created using files in RawData folder.
    private static String FEATUREFILEPATH = "Data/FeatureFiles/features_9.csv";
    
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
	    	//sum is the sum of all values of each sensor type
	        HashMap<String, Double> sum = new HashMap<String, Double>();
	        //avg is the average of all values of each sensor type
	        HashMap<String, Double> avg = new HashMap<String, Double>();
	        //var is the variance of all variances of each sensor type
	        HashMap<String, Double> var = new HashMap<String, Double>();
	        
	        
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
		    	        
		    	        //calculate the sum, average, and variance of all values for each feature
		    	        calculateSumAvgVar(values, sum, avg, var);
		    	        
		    	        //write feature column headers only once
		    	        if (row == 0) {
		    	        	writeFeatureFileHeaders(featureFileBuffer, sum, avg, var);
		    	        }
		    	        
		    	        //write single row of feature data, one column per feature
		    	        writeFeatureFile(featureFileBuffer, sum, avg, var, classification);
		    	        
		    	        //set all default values for each key to reuse containers with the same key ordering
		    	        for (String key : values.keySet()) {
		    	        	ArrayList<Double> tempArrayList = values.get(key);
		    	        	tempArrayList.clear();
		    	        	
		    	        	sum.replace(key, (double)0);
		    	        	avg.replace(key, (double)0);
		    	        	var.replace(key, (double)0);
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
    
    
    public static void calculateSumAvgVar(HashMap<String, ArrayList<Double>> values, HashMap<String, Double> sum, HashMap<String, Double> avg, HashMap<String, Double> var) {
        for (String key : values.keySet()) {
        	//calculate average and sum for each reading (value) for each sensor type (key)
        	for (double val : values.get(key)) {
        		if (sum.containsKey(key)) {
                	sum.put(key, sum.get(key) + val);
                } else {
                	sum.put(key, val);
                }
        	}
        	double keyAvg = sum.get(key) / values.get(key).size();
        	avg.put(key, keyAvg);
        	
        	//calculate variance for each reading (value) for each sensor type (key)
        	//variance is the average of the squared differences from the mean
        	for (double val : values.get(key)) {
        		if (var.containsKey(key)) {
        			var.put(key, var.get(key) + Math.pow(val - keyAvg, 2));
        		} else {
                   	var.put(key, Math.pow(val - keyAvg, 2));
                }
        	}
        	double keyVar = var.get(key) / values.get(key).size();
        	var.put(key, keyVar);
        }
    }
     
    //assumes values exist for sum, avg, and var
    public static void writeFeatureFileHeaders(BufferedWriter featureFileBuffer, HashMap<String, Double> sum, HashMap<String, Double> avg, HashMap<String, Double> var) {
		try {
			featureFileBuffer.write("classification,");
			
	    	//Write column headers.
			for (String key : sum.keySet()) {
				featureFileBuffer.write(key + "_sum,");
			}
			for (String key : avg.keySet()) {
				featureFileBuffer.write(key + "_avg,");
			}
			int i = 0;
			for (String key : var.keySet()) {
				i++;
				featureFileBuffer.write(key + "_var");
				
				//don't write comma after last feature
				if (i < var.keySet().size()) {
					featureFileBuffer.write(",");
				}
			}
			featureFileBuffer.newLine();
			featureFileBuffer.flush();
		} catch (IOException e) {
			System.err.println("Cannot write feature file headers.");
		}
    }
    
    public static void writeFeatureFile(BufferedWriter featureFileBuffer, HashMap<String, Double> sum, HashMap<String, Double> avg, HashMap<String, Double> var, String classification) {
    	try {
    		featureFileBuffer.write(classification + ",");
    		
	    	//Write all values for single row.
			for (String key : sum.keySet()) {
				double val = sum.get(key);
				featureFileBuffer.write(Double.toString(val) + ",");
				System.out.println("Sum: " + Double.toString(val));
			}
			for (String key : avg.keySet()) {
				double val = avg.get(key);
				featureFileBuffer.write(Double.toString(val) + ",");
				System.out.println("Avg: " + Double.toString(val));
			}
			int i = 0;
			for (String key : var.keySet()) {
				i++;
				double val = var.get(key);
				featureFileBuffer.write(Double.toString(val));
				System.out.println("Var: " + Double.toString(val));
				
				//don't write comma after last feature
				if (i < var.keySet().size()) {
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
        /*Random rn = new Random();
        for (int i = 0; i < 5; i++) {
            TEST_INDEX = rn.nextInt(15);
            
            System.out.println("Test instance is " + TEST_INDEX);
         */
    	
    	for (int i = dataSize; i > 0; i--) {
	    	System.out.println("---------------------------------------------");
	    	System.out.println("Number of Components = " + Integer.toString(i));
	    	
	    	REDUCED_LENGTH = i;
	        
	    	printNearestNeighbours(fullData, fullDataClassification, true);
	        printNearestNeighBoursPCA(fullData, fullDataClassification, dataAvg);
	            
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

    private static ArrayList<DistObj> printNearestNeighbours(ArrayList<double[]> fullData, ArrayList<String> fullDataClassification, boolean toprint) {
        int fullDataSize = fullData.size();
        int testIndex = fullDataSize - 1;

        ArrayList<DistObj> distObjects = new ArrayList<>();

        for (int i = 0; i < fullDataSize; i++) {
            double distances = calculateDistance(fullData.get(testIndex), fullData.get(i));
            DistObj dobj = new DistObj();
            dobj.index = i;
            dobj.distance = distances;
            distObjects.add(dobj);
        }

        sortDistObjs(distObjects);
        if(!toprint){return distObjects;}
        
        System.out.println("Neighbors Before PCA: \n");

        for (int i = 1; i <= NUM_OF_NEIGHBOURS; i++) {
            System.out.println("Neighbor " + i + ": Index=" + distObjects.get(i).index + ", Classification=" + fullDataClassification.get(distObjects.get(i).index) + ", Distance=" + distObjects.get(i).distance);
        }

        return distObjects;

    }

    private static void sortDistObjs(ArrayList<DistObj> distObjects) {
        Collections.sort(distObjects, new Comparator<DistObj>() {
            @Override
            public int compare(DistObj do1, DistObj do2) {
                return Double.compare(do1.distance, do2.distance);
            }
        });
    }

    private static void printNearestNeighBoursPCA(ArrayList<double[]> fullData, ArrayList<String> fullDataClassification, double[] dataAvg) {
        int fullDataSize = fullData.size();
        int testIndex = fullDataSize - 1;
        int dataSize = fullData.get(0).length;

        double[][] oldData2dArray = new double[fullDataSize][dataSize];

        int count = 0;

        for (double dataLine[] : fullData) {
            System.arraycopy(dataLine, 0, oldData2dArray[count], 0, dataSize);
            count++;
        }

        for (int i = 0; i < dataSize; i++) {
            dataAvg[i] /= fullDataSize;
        }

        // create a copy of fullData
        ArrayList<double[]> fullDataAdjust = new ArrayList<>(fullData);

        //creating data adjust
        for (double dataAdjustComps[] : fullDataAdjust) {
            for (int i = 0; i < dataSize; i++) {
                dataAdjustComps[i] -= dataAvg[i];
            }
        }

        double[][] covarianceMatrix = new double[dataSize][dataSize];

        for (int i = 0; i < dataSize; i++) {
            for (int j = 0; j < dataSize; j++) {
                covarianceMatrix[i][j] = calculateCovariance(fullDataAdjust, i, j);
            }
        }

        List<EigenObject> eigenObjList = performEigenOperations(covarianceMatrix, dataSize);
        int reducedDataSize = REDUCED_LENGTH;

        double[][] eigenVector2dArray = new double[dataSize][reducedDataSize];

        int eigenObjectCount = 0;

        for (EigenObject eigenObject : eigenObjList) {

            double[] eigenVector = eigenObject.getEigenVector();

            for (int i = 0; i < dataSize && eigenObjectCount < reducedDataSize; i++) {
                eigenVector2dArray[i][eigenObjectCount] = eigenVector[i];
            }

            eigenObjectCount++;
        }

        Matrix oldData = new Matrix(oldData2dArray);
        Matrix eigenVectors = new Matrix(eigenVector2dArray);

        Matrix newData = new Matrix(fullDataSize, dataSize);

        newData = oldData.times(eigenVectors);
        System.out.println("");

        double[][] newData2dArray = new double[fullDataSize][dataSize];

        newData2dArray = newData.getArrayCopy();
        ArrayList<double[]> fullNewData = new ArrayList<>();

        for (int i = 0; i < fullDataSize; i++) {
            fullNewData.add(newData2dArray[i]);
        }

        ArrayList<DistObj> distObjects = printNearestNeighbours(fullNewData, fullDataClassification, false);

        System.out.println("\nNeighbors After PCA: \n");

        for (int i = 1; i <= NUM_OF_NEIGHBOURS; i++) {
            System.out.println("Neighbor " + i + ": Index=" + distObjects.get(i).index + ", Classification=" + fullDataClassification.get(distObjects.get(i).index) + ", Distance=" + calculateDistance(fullNewData.get(testIndex), fullNewData.get(distObjects.get(i).index)));
        }
    }

    private static double calculateCovariance(ArrayList<double[]> fullDataAdjust, int i, int j) {

        double metricAdjustProdTotal = 0.0;  // the final numerator in the covariance formula i.e Summation[(Xi-Xmean)*(Yi-Ymean)]

        for (double dataAdjustComps[] : fullDataAdjust) {
            metricAdjustProdTotal += dataAdjustComps[i] * dataAdjustComps[j];
        }
        return metricAdjustProdTotal / (fullDataAdjust.size() - 1);
    }

    private static List<EigenObject> performEigenOperations(double[][] covarianceMatrix, int dataSize) {
        Matrix evdMatrix = new Matrix(covarianceMatrix);
        EigenvalueDecomposition evd = new EigenvalueDecomposition(evdMatrix);

        double[] myEigenValues = new double[dataSize];

        double[][] myEigenVectorMatrixInput = new double[dataSize][dataSize];
        Matrix myEigenVectorMatrix = new Matrix(myEigenVectorMatrixInput);

        myEigenValues = evd.getRealEigenvalues();
        myEigenVectorMatrix = evd.getV();
        
        List<EigenObject> eigenObjList = new ArrayList<>(dataSize);
        for (int i = 0; i < dataSize; i++) {
            eigenObjList.add(new EigenObject(myEigenValues[i], myEigenVectorMatrix.getArray()[i]));
        }

        Collections.sort(eigenObjList, new Comparator<EigenObject>() {
            @Override
            public int compare(EigenObject eo1, EigenObject eo2) {
                double eigenVal1 = eo1.getEigenValue();
                double eigenVal2 = eo2.getEigenValue();
                return (eigenVal1 == eigenVal2) ? 0 : (eigenVal1 < eigenVal2 ? 1 : -1);
            }
        });

        return eigenObjList;
    }
    
}