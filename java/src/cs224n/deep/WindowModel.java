package cs224n.deep;
import java.lang.*;
import java.util.*;
import java.io.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, Wout, U;

	private HashMap<String,String> exactMatches = new HashMap<String, String>();
	//
	public int windowSize,wordSize, hiddenSize;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		//TODO
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		//TODO
		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
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
		return W.mult(inputVector);
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
		SimpleMatrix gMat = U.mult(inputVector);
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
