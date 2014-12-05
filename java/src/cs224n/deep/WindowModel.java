package cs224n.deep;
import java.lang.*;
import java.util.*;
import java.io.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, Wout, U, FeatureMat;

	private HashMap<String,String> exactMatches = new HashMap<String, String>();
	//
	public int windowSize,wordSize, hiddenSize, H;
	public int K = 5;

	public int[][] matrix;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		//TODO
		windowSize = _windowSize;
		wordSize = 50;
		hiddenSize = _hiddenSize;
		H = _hiddenSize;
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(SimpleMatrix wordMat){
		//TODO
		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
		FeatureMat = wordMat;
		int lastIndex = wordSize*windowSize;
		W = SimpleMatrix.random(H,lastIndex + 1,0,1, new Random());
		for(int i = 0; i < H; i++){
			W.set(i, lastIndex, 1);
		}
		U = SimpleMatrix.random(K,H+1,0,1, new Random());
		for(int i = 0; i < K; i++){
			U.set(i,H,1);
		}
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		//	TODO
	}

	
	public void test(List<Datum> testData){
		// TODO
	}

	public void baselineTrain(List<Datum> _trainData ){
		for(Datum data: _trainData){
			exactMatches.put(data.word, data.label);
		}
	}


	public void baselineTest(List<Datum> testData){
		try{
			File f = new File("../baseline.out");
			PrintWriter writer = new PrintWriter("../baseline.out", "UTF-8");
			for(Datum test:testData){
				String guessedLabel = "O";
				if(exactMatches.containsKey(test.word)){
					guessedLabel = exactMatches.get(test.word);
				}
				writer.println(test.word + "\t" + test.label +"\t"+ guessedLabel);
			}
			writer.close();
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}

	private SimpleMatrix zFunction(SimpleMatrix inputVector){
		SimpleMatrix newVec = SimpleMatrix.random(inputVector.numRows(),1,1,1, new Random());
		newVec.insertIntoThis(0,0,inputVector);
		return W.mult(newVec);
	}

	private SimpleMatrix hFunction(SimpleMatrix inputVector){
		int numRows = inputVector.numRows();
		int numCols = inputVector.numCols();
		SimpleMatrix output = new SimpleMatrix(numRows, numCols);
		for(int index = 0; index < numRows*numCols; index++){
			output.set(index, Math.tanh(inputVector.get(index)));
		}
		return output;
	}

	private SimpleMatrix gFunction(SimpleMatrix inputVector){
		SimpleMatrix newVec = SimpleMatrix.random(inputVector.numRows(),1,1,1, new Random());
		newVec.insertIntoThis(0,0,inputVector);

		SimpleMatrix gMat = U.mult(newVec);
		int numRows = gMat.numRows();
		int numCols = gMat.numCols();
		int denom = 0;
		for(int index = 0; index < numCols*numRows; index++){
			denom += Math.exp(gMat.get(index));
		}
		for(int index = 0; index < numCols*numRows; index++){
			gMat.set(index, Math.exp(gMat.get(index))/denom);
		}
		return gMat;
	}

}
