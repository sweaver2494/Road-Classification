/*
 * This code makes a comparision of nearest neighbours (euclidean distance)
 * for randomly selected testinstances calculated before and after the 
 * application of PCA pre-processing.
 *
 * This code works for community dataset
 *
 * @author Nikhilesh pandey
 */

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Random;


public class KnnCompRoad {
    private static final int NUM_OF_NEIGHBOURS = 5;
    private static int TEST_INDEX = 10;
    private static final int REDUCED_LENGTH = 8;

    public static void main(String[] args) throws FileNotFoundException, IOException {
        //Contains each row of variables
        //ArrayList<double[]> fullData = new ArrayList<>();
        //Maps each row index in the data to its classification
        //HashMap<Integer, Double> indexClass = new HashMap<Integer, Double>();
        
        BufferedReader br = new BufferedReader(new FileReader("RawData/roads.csv"));
        //discard first line since it will contain headers
        br.readLine();
        
        
        //double dataAvg[] = new double[dataSize];
        
        //values contains the values of each sensor type
        HashMap<String, ArrayList<Double>> values = new HashMap<String, ArrayList<Double>>();
        //sum is the sum of all values of each sensor type
        HashMap<String, Double> sum = new HashMap<String, Double>();
        //avg is the average of all values of each sensor type
        HashMap<String, Double> avg = new HashMap<String, Double>();
        //var is the variance of all variances of each sensor type
        HashMap<String, Double> var = new HashMap<String, Double>();
        
        //int index = 0;

        

        String line = br.readLine();
        while (line != null) {
        	String dataCompsStr[] = line.split(",");
        	String key = dataCompsStr[1];
        	double val = Double.parseDouble(dataCompsStr[3]);
        	
        	if (values.containsKey(key)) {
        		ArrayList<Double> temp = values.get(key);
        		temp.add(val);
        	} else {
        		ArrayList<Double> temp = new ArrayList<Double>();
        		temp.add(val);
        		values.put(key, temp);
        	}
        	
        	line = br.readLine();
        }
        
      
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

            //double dataComps[] = new double[dataSize];
            
            //dataComps[i] = Double.parseDouble(dataCompsStr[i]);
            //dataAvg[i] += dataComps[i];
            
            //Add the row of variables to the data set
            //fullData.add(dataComps);
            
            //The last column is not a variable, but instead specifies the classification.
            //indexClass hashes the index of each row to its classification.
            //double classification = Double.parseDouble(dataCompsStr[dataSize]);
            //indexClass.put(index, classification);
        	//index++;
            
        //}
        br.close();
        
        //testPCA(fullData, indexClass, dataAvg);
        //testVars(fullData, indexClass, dataAvg);


    }
    
    public static void testPCA(ArrayList<double[]> fullData, HashMap<Integer,Double> indexClass, double[] dataAvg) {
        Random rn = new Random();
        for (int i = 0; i < 5; i++) {
            TEST_INDEX = rn.nextInt(13);
            
            System.out.println("Test instance is " + TEST_INDEX);

            printNearestNeighbours(fullData, indexClass, true);
            printNearestNeighBoursPCA(fullData, indexClass, dataAvg);
            
            System.out.println("---------------------------------------------");

        }
    }
    
    public static void testVars(ArrayList<double[]> fullData, HashMap<Integer,Double> indexClass, double[] dataAvg) {
    	
    }

    public static double calculateDistance(double[] array1, double[] array2) {
        double Sum = 0.0;
        for (int i = 0; i < array1.length; i++) {
            Sum = Sum + Math.pow((array1[i] - array2[i]), 2.0);
        }
        return Math.sqrt(Sum);
    }

    private static ArrayList<DistObj> printNearestNeighbours(ArrayList<double[]> fullData, HashMap<Integer,Double> indexClass, boolean toprint) {
        int testIndex = TEST_INDEX;
        int fullDataSize = fullData.size();

        

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
            System.out.println("Neighbor " + i + ": Index=" + distObjects.get(i).index + ", Classification=" + indexClass.get(distObjects.get(i).index) + ", Distance=" + distObjects.get(i).distance);
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

    private static void printNearestNeighBoursPCA(ArrayList<double[]> fullData, HashMap<Integer,Double> indexClass, double[] dataAvg) {
        int fullDataSize = fullData.size();
        int dataSize = fullData.get(0).length;
        //double dataAvg[] = new double[dataSize];

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

        ArrayList<DistObj> distObjects = printNearestNeighbours(fullNewData, indexClass, false);

        System.out.println("\nNeighbors After PCA: \n");

        for (int i = 1; i <= NUM_OF_NEIGHBOURS; i++) {
            System.out.println("Neighbor " + i + ": Index=" + distObjects.get(i).index + ", Classification=" + indexClass.get(distObjects.get(i).index) + ", Distance=" + calculateDistance(fullNewData.get(TEST_INDEX), fullNewData.get(distObjects.get(i).index)));
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