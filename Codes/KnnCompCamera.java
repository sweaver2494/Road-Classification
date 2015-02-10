/*
 * This code makes a comparision of nearest neighbours (euclidean distance)
 * for randomly selected testinstances calculated before and after the 
 * application of PCA pre-processing.
 *
 * This code works for camera dataset
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
import java.util.List;
import java.util.Random;


public class KnnCompCamera {
    private static final int NUM_OF_NEIGHBOURS = 5;
    private static int TEST_INDEX = 500;
    private static final int REDUCED_LENGTH = 10;

    public static void main(String[] args) throws FileNotFoundException, IOException {
        Random rn = new Random();
        ArrayList<double[]> fullData = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader("RawData/CameraData.csv"));
        String line = br.readLine();

        int dataSize = line.length() - line.replace(",", "").length() + 1;

        double dataAvg[] = new double[dataSize];
        int ct = 0;

        while (line != null) {
            String dataCompsStr[] = line.split(",");

            double dataComps[] = new double[dataSize];

            for (int i = 0; i < dataSize; i++) {
                dataComps[i] = Double.parseDouble(dataCompsStr[i]);
                dataAvg[i] += dataComps[i];
            }

            fullData.add(dataComps);
            line = br.readLine();
        }

        for (int i = 0; i < 5; i++) {
            TEST_INDEX = rn.nextInt(1000);
            
            System.out.println("Test instance is " + TEST_INDEX);

            printNearestNeighbours(fullData, true);
            printNearestNeighBoursPCA(fullData);
            
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

    private static ArrayList<DistObj> printNearestNeighbours(ArrayList<double[]> fullData, boolean toprint) {
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
        
        System.out.println("Nearest neighbours before pre processing : \n");

        for (int i = 1; i <= NUM_OF_NEIGHBOURS; i++) {
            System.out.println("Neighbours number " + i + " = Index : " + distObjects.get(i).index + " Distance : " + distObjects.get(i).distance);
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

    private static void printNearestNeighBoursPCA(ArrayList<double[]> fullData) {
        int fullDataSize = fullData.size();
        int dataSize = fullData.get(0).length;
        double dataAvg[] = new double[dataSize];

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

        ArrayList<DistObj> distObjects = printNearestNeighbours(fullNewData, false);

        System.out.println("\nNearest neighbours after PCA pre processing : \n");

        for (int i = 1; i <= NUM_OF_NEIGHBOURS; i++) {
            System.out.println("Neighbours number " + i + " = Index : " + distObjects.get(i).index + " Distance : " + calculateDistance(fullData.get(TEST_INDEX), fullData.get(distObjects.get(i).index)));
        }
    }

    private static double calculateCovariance(ArrayList<double[]> fullDataAdjust, int i, int j) {

        double metricAdjustProdTotal = 0.0;  // the final numerator in the covariance formula i.e Summation[(Xi-Xmean)*(Yi-Ymean)]

        for (double dataAdjustComps[] : fullDataAdjust) {
            metricAdjustProdTotal += dataAdjustComps[i] * dataAdjustComps[j];
        }
        return metricAdjustProdTotal / (fullDataAdjust.size() - 1); 
    }

    private static List performEigenOperations(double[][] covarianceMatrix, int dataSize) {
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
