import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import javafx.util.Pair;
import Jama.EigenvalueDecomposition;
import Jama.Matrix;


public class MLUtilities {
	
	public static ArrayList<DistObj> performKNN(ArrayList<double[]> fullData, double[] testData) {
        int fullDataSize = fullData.size();

        ArrayList<DistObj> distObjects = new ArrayList<>();

        for (int i = 0; i < fullDataSize; i++) {
            double distances = calculateDistance(testData, fullData.get(i));
            DistObj dobj = new DistObj();
            dobj.index = i;
            dobj.distance = distances;
            distObjects.add(dobj);
        }

        sortDistObjs(distObjects);
        return distObjects;
	}
	
    public static double calculateDistance(double[] array1, double[] array2) {
        double Sum = 0.0;
        for (int i = 0; i < array1.length; i++) {
            Sum = Sum + Math.pow((array1[i] - array2[i]), 2.0);
        }
        return Math.sqrt(Sum);
    }
    
    private static void sortDistObjs(ArrayList<DistObj> distObjects) {
        Collections.sort(distObjects, new Comparator<DistObj>() {
            @Override
            public int compare(DistObj do1, DistObj do2) {
                return Double.compare(do1.distance, do2.distance);
            }
        });
    }
    
    public static Pair<ArrayList<double[]>, double[]> performPCA(ArrayList<double[]> fullData, double[] testDataArr, double[] dataAvg, int reducedDataSize) {
        
    	//find eigenvectors of data set
    	Matrix eigenVectors = getEigenVectors(fullData, dataAvg, reducedDataSize);
        
    	//use eigenvectors to calculate the reduced data set
        ArrayList<double[]> fullNewData = calculatePCA(fullData, eigenVectors);
        
        //use the same eigenvectors to reduce the test data to fit in the same dimensionality as the training data
        ArrayList<double[]> testData = new ArrayList<>();
        testData.add(testDataArr);
        double[] testNewData = calculatePCA(testData, eigenVectors).get(0);
        
        return new Pair<ArrayList<double[]>, double[]>(fullNewData, testNewData);
    }
    
    private static ArrayList<double[]> calculatePCA(ArrayList<double[]> oldDataList, Matrix eigenVectors) {
    	int oldDataSize = oldDataList.size();
        int dataSize = oldDataList.get(0).length;
        
        double[][] oldData2dArray = new double[oldDataSize][dataSize];

        int count = 0;
        for (double dataLine[] : oldDataList) {
            System.arraycopy(dataLine, 0, oldData2dArray[count], 0, dataSize);
            count++;
        }
        
        Matrix oldDataMatrix = new Matrix(oldData2dArray);
        Matrix newDataMatrix = new Matrix(oldDataSize, dataSize);
        
        newDataMatrix = oldDataMatrix.times(eigenVectors);

        double[][] newData2dArray = new double[oldDataSize][dataSize];
        
        newData2dArray = newDataMatrix.getArrayCopy();
        ArrayList<double[]> newDataList = new ArrayList<>();

        for (int i = 0; i < oldDataSize; i++) {
            newDataList.add(newData2dArray[i]);
        }
        
        return newDataList;
    }
    
    private static Matrix getEigenVectors(ArrayList<double[]> fullData, double[] dataAvg, int reducedDataSize) {
        int dataSize = fullData.get(0).length;
    	
    	// create a copy of fullData
        ArrayList<double[]> fullDataAdjust = new ArrayList<>();
        for (double[] src : fullData) {
        	double[] dest = new double[dataSize];
        	System.arraycopy( src, 0, dest, 0, src.length );
        	
        	fullDataAdjust.add(dest);
        }
        
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

        double[][] eigenVector2dArray = new double[dataSize][reducedDataSize];

        int eigenObjectCount = 0;

        for (EigenObject eigenObject : eigenObjList) {

            double[] eigenVector = eigenObject.getEigenVector();

            for (int i = 0; i < dataSize && eigenObjectCount < reducedDataSize; i++) {
                eigenVector2dArray[i][eigenObjectCount] = eigenVector[i];
            }

            eigenObjectCount++;
        }
        
        Matrix eigenVectors = new Matrix(eigenVector2dArray);
        
        return eigenVectors;
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
