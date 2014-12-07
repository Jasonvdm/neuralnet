package cs224n.deep;
import java.lang.*;
import java.util.*;
import java.io.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, Wout, U, XMatrix, YMatrix;

	private HashMap<String,String> exactMatches = new HashMap<String, String>();
	HashMap<String, Integer> wordNum;
	HashMap<String, Integer> convertLabelToInt = new HashMap<String, Integer>();;
	HashMap<Integer,String> convertIntToLabel = new HashMap<Integer, String>();;
	//
	public int windowSize,wordSize, hiddenSize, H;
	public int K = 5;
	public int nC;
	public int m;

	public int[][] matrix;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		//TODO
		windowSize = _windowSize;
		wordSize = 50;
		nC = wordSize*windowSize;
		hiddenSize = _hiddenSize;
		H = _hiddenSize;
		convertIntToLabel.put(0,"O");
		convertIntToLabel.put(1,"MISC");
		convertIntToLabel.put(2,"PER");
		convertIntToLabel.put(3,"ORG");
		convertIntToLabel.put(4,"LOC");
		convertLabelToInt.put("O",0);
		convertLabelToInt.put("MISC",1);
		convertLabelToInt.put("PER",2);
		convertLabelToInt.put("ORG",3);
		convertLabelToInt.put("LOC",4);
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(SimpleMatrix wordMat, HashMap<String, Integer> wordToNum){
		//TODO
		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
		L = wordMat;
		wordNum = wordToNum;
		W = SimpleMatrix.random(H,nC + 1,-Math.sqrt(6)/Math.sqrt(nC + H),Math.sqrt(6)/Math.sqrt(nC + H), new Random());
		for(int i = 0; i < H; i++){
			W.set(i, nC, 1);
		}
		U = SimpleMatrix.random(K,H+1,-Math.sqrt(6)/Math.sqrt(K + H),Math.sqrt(6)/Math.sqrt(K + H), new Random());
		for(int i = 0; i < K; i++){
			U.set(i,H,1);
		}
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		//	TODO
		m = 100;//_trainData.size();
		XMatrix = new SimpleMatrix(nC,m);
		YMatrix = new SimpleMatrix(K,m);
		for(int index = 0; index < 100; index++){
			//System.out.println(_trainData.get(index).word+"\t"+_trainData.get(index).label);
			String currentWord = _trainData.get(index).word;
			SimpleMatrix xVector = new SimpleMatrix(nC,1);
			SimpleMatrix yVector = new SimpleMatrix(K,1);
			int[] wordNums = new int[windowSize];
			int startToken = wordNum.get("<s>");
			int endToken = wordNum.get("</s>");
			boolean hitDocStart = false;
			if(_trainData.get(index).word.equals("-DOCSTART-")){
				continue;
			}
			yVector.set(convertLabelToInt.get(_trainData.get(index).label),0,1);
			wordNums[windowSize/2] = getWordsNumber(currentWord);

			for(int wordIndex = index-1; wordIndex >= index - (windowSize/2); wordIndex--){ 
				if(!hitDocStart && _trainData.get(wordIndex).word.equals("-DOCSTART-")) hitDocStart = true;
				if(hitDocStart) wordNums[wordIndex - index + (windowSize/2)] = startToken;
				else {
					currentWord = _trainData.get(wordIndex).word; 
					wordNums[wordIndex - index + (windowSize/2)] = getWordsNumber(currentWord);
				}
			}

			hitDocStart = false;
			for(int wordIndex = index+1; wordIndex <= index + (windowSize/2); wordIndex++){
				if(wordIndex >= _trainData.size() || _trainData.get(wordIndex).word.equals("-DOCSTART-")) hitDocStart = true;
				if(hitDocStart) wordNums[wordIndex - index + (windowSize/2)] = endToken;
				else {
					currentWord = _trainData.get(wordIndex).word; 
					wordNums[wordIndex - index + (windowSize/2)] = getWordsNumber(currentWord);
				}
			}

			for(int wordIndex = 0; wordIndex < windowSize; wordIndex++){
				int i = wordNums[wordIndex];
				for(int j = 0; j < wordSize; j++){
					xVector.set(wordIndex*wordSize+j,0,L.get(i,j));
				}
			}
			XMatrix.insertIntoThis(0,index,xVector);
			YMatrix.insertIntoThis(0,index,yVector);
		}
		XMatrix.print("%e");
		gradientCheck();
	}

	private int getWordsNumber(String word){
		String lowerWord = word.toLowerCase();
		if(wordNum.containsKey(lowerWord)) return wordNum.get(lowerWord);
		return wordNum.get("UUUNKKK");
	}

	
	public void test(List<Datum> testData){
		// TODO
	}

	public void baselineTrain(List<Datum> _trainData ){
		for(Datum data: _trainData){
			exactMatches.put(data.word, data.label);
		}
	}


	public double costFunction(SimpleMatrix inputVector, SimpleMatrix trueLabel){
		double cost = 0;
		double lTotal = 0;
		for (int j = 0; j < K; j++) {
			lTotal += trueLabel.get(j) * Math.log(gFunction(inputVector).get(j));
		}
		return (-1 * lTotal);
	}

	public double regularizedCostFunction(SimpleMatrix input, SimpleMatrix label) {
		double costF = costFunction(input, label);

		double lambda = 1.0;
		int nC = 50 * 3;

		double wSum = 0;
		double uSum = 0;
		for (int i=0; i < H; i++) {
			for (int j=0; j < nC; j++) {
				wSum += (W.get(i, j) * W.get(i, j));
			}
		}

		for (int i=0; i < K; i++) {
			for (int j=0; j < H; j++) {
				uSum += (U.get(i, j) * U.get(i, j));
			}
		}

		return costF + (lambda * (wSum + uSum))/(2*m);
	}

	public void gradientCheck() {
		double epsilon = 0.0001;
		for(int index = 0; index < m; index++){
			ArrayList diffVector = new ArrayList<Double>();
			SimpleMatrix xVector = new SimpleMatrix(nC,1);
			SimpleMatrix yVector = new SimpleMatrix(K,1);
			for(int i = 0; i < nC; i++){
				xVector.set(i,0, XMatrix.get(i,index));
			}
			for(int i = 0; i < K; i++){
				yVector.set(i,0, YMatrix.get(i,index));
			}
		// Change L by epsilon first
		// SimpleMatrix theta = new SimpleMatrix(L);
		// for (int i = 0; i < theta.numRows(); i++){
		// 	for (int j = 0; j < theta.numCols(); j++){
		// 		theta.set(i, j, theta.get(i,j) + epsilon);
		// 		double jPlus = costFunction(xVector, yLabels);
		// 		theta.set(i, j, theta.get(i,j) - 2*epsilon);
		// 		double jMinus = costFunction(xVector, yLabels);
		// 		double costDiff = (jPlus - jMinus)/2*epsilon;
		// 		double gradientVal = uGradient(i,j, xVector, yLabels);
		// 		diffVector.add(gradientVal - costDiff);
		// 	}
		// }

		// theta = new SimpleMatrix(W);
		// for (int i = 0; i < theta.numRows(); i++){
		// 	for (int j = 0; j < theta.numCols(); j++){
		// 		theta.set(i, j, theta.get(i,j) + epsilon);
		// 		double jPlus = costFunction(xVector, yLabels);
		// 		theta.set(i, j, theta.get(i,j) - 2*epsilon);
		// 		double jMinus = costFunction(xVector, yLabels);
		// 		double costDiff = (jPlus - jMinus)/2*epsilon;
		// 		double gradientVal = uGradient(i,j, xVector, yLabels);
		// 		diffVector.add(gradientVal - costDiff);
		// 	}
		// }

		//System.out.println("U: "+U.numRows()+"x"+U.numCols());
			for (int i = 0; i < U.numRows(); i++){
				for (int j = 0; j < U.numCols(); j++){
					U.set(i, j, U.get(i,j) + epsilon);
					double jPlus = costFunction(xVector,yVector);
					U.set(i, j, U.get(i,j) - 2*epsilon);
					double jMinus = costFunction(xVector,yVector);
					//System.out.println(jPlus +" : "+ jMinus);
					double costDiff = (jPlus - jMinus)/(2*epsilon);
					U.set(i, j, U.get(i,j) + epsilon);
					double gradientVal = uGradient(i,j, xVector, yVector);
					System.out.println(gradientVal +" : "+ costDiff);
					//System.out.println(gradientVal - costDiff);
					diffVector.add(gradientVal - costDiff);
				}
			}
			
			double gradientDifference = normalizeVector(diffVector);
			System.out.println("Gradient difference is: " + gradientDifference);
		}
	}

	private double normalizeVector(ArrayList<Double> list) {
		double total = 0;

		for (int i = 0; i < list.size(); i++) {
			Double v = list.get(i);
			double val = v.doubleValue();
			total += val*val;
		}
		total = Math.sqrt(total);
		return total;
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
		SimpleMatrix newVec = SimpleMatrix.random(inputVector.numRows()+1,1,1,1, new Random());
		newVec.insertIntoThis(0,0,inputVector);
		//System.out.println("W: "+W.numRows()+"x"+W.numCols() + "\tX: "+newVec.numRows()+"x"+newVec.numCols());
		return W.mult(newVec);
	}

	private SimpleMatrix hFunction(SimpleMatrix inputVector){
		int numRows = inputVector.numRows();
		SimpleMatrix output = new SimpleMatrix(numRows, 1);
		for(int index = 0; index < numRows; index++){
			output.set(index,0, Math.tanh(inputVector.get(index,0)));
		}
		return output;
	}

	private SimpleMatrix gFunction(SimpleMatrix inputVector){
		SimpleMatrix inputVec = hFunction(zFunction(inputVector));
		SimpleMatrix newVec = SimpleMatrix.random(inputVec.numRows()+1,1,1,1, new Random());
		newVec.insertIntoThis(0,0,inputVec);

		SimpleMatrix gMat = U.mult(newVec);
		double denom = 0;
		for(int index = 0; index < K; index++){
			denom += Math.exp(gMat.get(index,0));
		}
		for(int index = 0; index < K; index++){
			gMat.set(index,0, Math.exp(gMat.get(index))/denom);
		}
		return gMat;
	}

	private double uGradient(int i, int j, SimpleMatrix xVector, SimpleMatrix yVector){
			SimpleMatrix delta2 = yVector.minus(gFunction(xVector));
			SimpleMatrix atemp = hFunction(zFunction(xVector));
			SimpleMatrix a = SimpleMatrix.random(atemp.numRows()+1,1,1,1, new Random());
			a.insertIntoThis(0,0,atemp);
		//System.out.println("delta2: "+delta2.numRows()+"x"+delta2.numCols() + "\ta: "+a.numRows()+"x"+a.numCols());
		return -1*delta2.get(i)*a.get(j);
	}

	private double wGradient(int i, int j, SimpleMatrix inputVector, SimpleMatrix trueLabel){
		SimpleMatrix delta2 = trueLabel.minus(gFunction(inputVector));
		double out = calculateWOutput(delta2, j);
		return out;
	}

	private double calculateWOutput(SimpleMatrix delta2, int j){
		double output = 0;
		for(int i = 0; i < K; i ++){
			output += delta2.get(i)*U.get(i,j);
		}
		return output;
	}

}