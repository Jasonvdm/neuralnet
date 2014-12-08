package cs224n.deep;
import java.lang.*;
import java.util.*;
import java.io.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, Wout, U, XMatrix, YMatrix, testXMatrix, b1, b2, LGrad, WGrad, UGrad, b1Grad, b2Grad;

	private HashMap<String,String> exactMatches = new HashMap<String, String>();
	HashMap<String, Integer> wordNum;
	HashMap<String, Integer> convertLabelToInt = new HashMap<String, Integer>();;
	HashMap<Integer,String> convertIntToLabel = new HashMap<Integer, String>();;
	//
	public int windowSize,wordSize, hiddenSize, H;
	public int K = 5;
	public int nC;
	public int m;
	public double learningRate;
	public double lambda;

	public int[][] matrix;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr, double _lambda){
		//TODO
		System.out.println(_windowSize + " : " +_hiddenSize + " : " +_lr + " : " + _lambda);
		lambda = _lambda;
		windowSize = _windowSize;
		wordSize = 50;
		nC = wordSize*windowSize;
		hiddenSize = _hiddenSize;
		H = _hiddenSize;
		learningRate = _lr;
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
		W = SimpleMatrix.random(H,nC,-Math.sqrt(6)/Math.sqrt(nC + H),Math.sqrt(6)/Math.sqrt(nC + H), new Random());
		U = SimpleMatrix.random(K,H,-Math.sqrt(6)/Math.sqrt(K + H),Math.sqrt(6)/Math.sqrt(K + H), new Random());
		b1 = new SimpleMatrix(H,1);
		b2 = new SimpleMatrix(K,1);
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		//	TODO
		m = _trainData.size();
		XMatrix = new SimpleMatrix(windowSize,m);
		YMatrix = new SimpleMatrix(K,m);
		for(int index = 0; index < m; index++){
			//System.out.println(_trainData.get(index).word+"\t"+_trainData.get(index).label);
			String currentWord = _trainData.get(index).word;
			SimpleMatrix xVector = new SimpleMatrix(nC,1);
			SimpleMatrix yVector = new SimpleMatrix(K,1);
			int[] wordNums = new int[windowSize];
			int startToken = wordNum.get("<s>");
			int endToken = wordNum.get("</s>");
			boolean hitDocStart = false;
			if(_trainData.get(index).word.equals("-DOCSTART-")){
				hitDocStart = true;
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
			if(_trainData.get(index).word.equals("-DOCSTART-")){
				hitDocStart = true;
			}
			for(int wordIndex = index+1; wordIndex <= index + (windowSize/2); wordIndex++){
				if(wordIndex >= _trainData.size() || _trainData.get(wordIndex).word.equals("-DOCSTART-")) hitDocStart = true;
				if(hitDocStart) wordNums[wordIndex - index + (windowSize/2)] = endToken;
				else {
					currentWord = _trainData.get(wordIndex).word; 
					wordNums[wordIndex - index + (windowSize/2)] = getWordsNumber(currentWord);
				}
			}
			for(int wordNumIndex = 0; wordNumIndex<windowSize; wordNumIndex++){
				XMatrix.set(wordNumIndex,index,wordNums[wordNumIndex]);
			}

			YMatrix.insertIntoThis(0,index,yVector);
		}
		//gradientCheck();
		runSGD();

	}

	private void runSGD(){
		int iters = 10;
		for (int step = 0; step < iters; step++){
			System.out.println(step);
			for(int exampleNum = 0; exampleNum < m; exampleNum++){
				SimpleMatrix xVector = new SimpleMatrix(nC,1);
				SimpleMatrix yVector = new SimpleMatrix(K,1);
				for(int wordIndex = 0; wordIndex < windowSize; wordIndex++){
					int i = (int)XMatrix.get(wordIndex,exampleNum);
					for(int j = 0; j < wordSize; j++){
						xVector.set(wordIndex*wordSize+j,0,L.get(i,j));
					}
				}
				for(int i = 0; i < K; i++){
					yVector.set(i,0, YMatrix.get(i,exampleNum));
				}
				uRegGradient(xVector, yVector);
				wRegGradient(xVector, yVector);
				lGradient(xVector, yVector);
				b1Gradient(xVector, yVector);
				b2Gradient(xVector, yVector);
				updateU();
				updateW();
				updateL(exampleNum);
				updateB1();
				updateB2();
			}
		}
	}


	private void updateU(){
		U = U.minus(UGrad.scale(learningRate));
	}

	private void updateW(){
		W = W.minus(WGrad.scale(learningRate));	
	}

	private void updateL(int exampleNum){
		for(int index = 0; index < windowSize; index++){
			int updateIndex = (int)XMatrix.get(index,exampleNum);
			for(int singleWordIndex = 0; singleWordIndex < wordSize; singleWordIndex++){
				double oldVal = L.get(updateIndex,singleWordIndex);
				L.set(updateIndex,singleWordIndex,oldVal - learningRate*LGrad.get(index*wordSize+singleWordIndex,0));
			}
		}
	}

	private void updateB1(){
		b1 = b1.minus(b1Grad.scale(learningRate));	
	}

	private void updateB2(){
		b2 = b2.minus(b2Grad.scale(learningRate));
	}

	private int getWordsNumber(String word){
		String lowerWord = word.toLowerCase();
		if(wordNum.containsKey(lowerWord)) return wordNum.get(lowerWord);
		return wordNum.get("UUUNKKK");
	}

	
	public void test(List<Datum> testData, String fileName){
		testXMatrix = new SimpleMatrix(windowSize,testData.size());
		for(int index = 0; index < testData.size(); index++){
			//System.out.println(testData.get(index).word+"\t"+testData.get(index).label);
			String currentWord = testData.get(index).word;
			SimpleMatrix xVector = new SimpleMatrix(nC,1);
			int[] wordNums = new int[windowSize];
			int startToken = wordNum.get("<s>");
			int endToken = wordNum.get("</s>");
			boolean hitDocStart = false;
			if(testData.get(index).word.equals("-DOCSTART-")){
				hitDocStart = true;
			}
			wordNums[windowSize/2] = getWordsNumber(currentWord);

			for(int wordIndex = index-1; wordIndex >= index - (windowSize/2); wordIndex--){ 
				if(!hitDocStart && testData.get(wordIndex).word.equals("-DOCSTART-")) hitDocStart = true;
				if(hitDocStart) wordNums[wordIndex - index + (windowSize/2)] = startToken;
				else {
					currentWord = testData.get(wordIndex).word; 
					wordNums[wordIndex - index + (windowSize/2)] = getWordsNumber(currentWord);
				}
			}

			hitDocStart = false;
			if(testData.get(index).word.equals("-DOCSTART-")){
				hitDocStart = true;
			}
			for(int wordIndex = index+1; wordIndex <= index + (windowSize/2); wordIndex++){
				if(wordIndex >= testData.size() || testData.get(wordIndex).word.equals("-DOCSTART-")) hitDocStart = true;
				if(hitDocStart) wordNums[wordIndex - index + (windowSize/2)] = endToken;
				else {
					currentWord = testData.get(wordIndex).word; 
					wordNums[wordIndex - index + (windowSize/2)] = getWordsNumber(currentWord);
				}
			}
			for(int wordNumIndex = 0; wordNumIndex<windowSize; wordNumIndex++){
				testXMatrix.set(wordNumIndex,index,wordNums[wordNumIndex]);
			}
		}
		try{
			File f = new File("neural.out");
			PrintWriter writer = new PrintWriter("neural.out", "UTF-8");
			for(int exampleNum = 0; exampleNum < testData.size(); exampleNum++){
				SimpleMatrix xVector = new SimpleMatrix(nC,1);
				for(int wordIndex = 0; wordIndex < windowSize; wordIndex++){
					int i = (int)testXMatrix.get(wordIndex,exampleNum);
					for(int j = 0; j < wordSize; j++){
						xVector.set(wordIndex*wordSize+j,0,L.get(i,j));
					}
				}
				String guessedLabel = predictLabel(xVector);
				writer.println(testData.get(exampleNum).word + "\t" + testData.get(exampleNum).label +"\t"+ guessedLabel);
			}
			writer.close();
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
	private String predictLabel(SimpleMatrix xVector){
		SimpleMatrix g = pFunction(xVector);
		int bestIndex = 0;
		double bestVal = g.get(0,0);
		for(int i = 1; i < K; i++){
			double value = g.get(i,0);
			if(value > bestVal){
				bestVal = value;
				bestIndex = i;
			}
		}
		return convertIntToLabel.get(bestIndex);
	}

	public void baselineTrain(List<Datum> _trainData ){
		for(Datum data: _trainData){
			exactMatches.put(data.word, data.label);
		}
	}


	public double costFunction(SimpleMatrix inputVector, SimpleMatrix trueLabel){
		double lTotal = 0;
		for (int j = 0; j < K; j++) {
			lTotal += trueLabel.get(j) * Math.log(pFunction(inputVector).get(j));
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
			for(int wordIndex = 0; wordIndex < windowSize; wordIndex++){
				int i = (int)XMatrix.get(wordIndex,index);
				for(int j = 0; j < wordSize; j++){
					xVector.set(wordIndex*wordSize+j,0,L.get(i,j));
				}
			}
			for(int i = 0; i < K; i++){
				yVector.set(i,0, YMatrix.get(i,index));
			}
			b1Gradient(xVector,yVector);
			b2Gradient(xVector,yVector);
			uGradient(xVector,yVector);
			wGradient(xVector,yVector);
			
			lGradient(xVector,yVector);
			// Change L by epsilon first
			for (int i = 0; i < xVector.numRows(); i++){
				for (int j = 0; j < xVector.numCols(); j++){
					xVector.set(i, j, xVector.get(i,j) + epsilon);
					double jPlus = costFunction(xVector, yVector);
					xVector.set(i, j, xVector.get(i,j) - 2*epsilon);
					double jMinus = costFunction(xVector, yVector);
					double costDiff = (jPlus - jMinus)/(2*epsilon);
					xVector.set(i, j, xVector.get(i,j) + epsilon);
					double gradientVal = LGrad.get(i,j);
					diffVector.add(gradientVal - costDiff);
				}
			}

			for (int i = 0; i < W.numRows(); i++){
				for (int j = 0; j < W.numCols(); j++){
					W.set(i, j, W.get(i,j) + epsilon);
					double jPlus = costFunction(xVector, yVector);
					W.set(i, j, W.get(i,j) - 2*epsilon);
					double jMinus = costFunction(xVector, yVector);
					double costDiff = (jPlus - jMinus)/(2*epsilon);
					W.set(i, j, W.get(i,j) + epsilon);
					double gradientVal = WGrad.get(i,j);
					diffVector.add(gradientVal - costDiff);
				}
			}
			for(int i = 0; i < b1.numRows(); i ++){
				b1.set(i,0,b1.get(i,0)+epsilon);
				double jPlus = costFunction(xVector,yVector);
				b1.set(i, 0, b1.get(i,0) - 2*epsilon);
				double jMinus = costFunction(xVector,yVector);
				double costDiff = (jPlus - jMinus)/(2*epsilon);
				b1.set(i, 0, b1.get(i,0) + epsilon);
				double gradientVal = b1Grad.get(i,0);
				diffVector.add(gradientVal - costDiff);
			}
			for(int i = 0; i < b2.numRows(); i ++){
				b2.set(i,0,b2.get(i,0)+epsilon);
				double jPlus = costFunction(xVector,yVector);
				b2.set(i, 0, b2.get(i,0) - 2*epsilon);
				double jMinus = costFunction(xVector,yVector);
				double costDiff = (jPlus - jMinus)/(2*epsilon);
				b2.set(i, 0, b2.get(i,0) + epsilon);
				double gradientVal = b2Grad.get(i,0);
				diffVector.add(gradientVal - costDiff);
			}
			for (int i = 0; i < U.numRows(); i++){
				for (int j = 0; j < U.numCols(); j++){
					U.set(i, j, U.get(i,j) + epsilon);
					double jPlus = costFunction(xVector,yVector);
					U.set(i, j, U.get(i,j) - 2*epsilon);
					double jMinus = costFunction(xVector,yVector);
					double costDiff = (jPlus - jMinus)/(2*epsilon);
					U.set(i, j, U.get(i,j) + epsilon);
					double gradientVal = UGrad.get(i,j);
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

	private SimpleMatrix zFunction(SimpleMatrix xVector){
		return W.mult(xVector).plus(b1);
	}

	private SimpleMatrix aFunction(SimpleMatrix xVector){
		SimpleMatrix z = zFunction(xVector);
		for(int index = 0; index < z.numRows(); index++){
			z.set(index,0,Math.tanh(z.get(index,0)));
		}
		return z;
	}

	private SimpleMatrix pFunction(SimpleMatrix xVector){
		return softMax(U.mult(aFunction(xVector)).plus(b2));
	}

	private SimpleMatrix softMax(SimpleMatrix g){
		double denom = 0;
		SimpleMatrix softMaxG = new SimpleMatrix(g);
		for(int index = 0; index < g.numRows(); index++){
			denom += Math.exp(g.get(index,0));
		}
		for(int index = 0; index < g.numRows(); index++){
			softMaxG.set(index,0,Math.exp(g.get(index,0))/denom);
		}
		return softMaxG;
	}


	private void uGradient(SimpleMatrix xVector, SimpleMatrix yVector){
		SimpleMatrix delta2 = pFunction(xVector).minus(yVector);
		SimpleMatrix a = aFunction(xVector);
		UGrad = delta2.mult(a.transpose());
	}

	private void uRegGradient(SimpleMatrix xVector, SimpleMatrix yVector){
		SimpleMatrix delta2 = pFunction(xVector).minus(yVector);
		SimpleMatrix a = aFunction(xVector);
		UGrad = delta2.mult(a.transpose()).plus(U.scale(lambda));
	}

	private void b2Gradient(SimpleMatrix xVector, SimpleMatrix yVector){
		b2Grad = pFunction(xVector).minus(yVector);
	}

	private void wGradient(SimpleMatrix xVector, SimpleMatrix trueLabel){
		SimpleMatrix delta2 = pFunction(xVector).minus(trueLabel);
		SimpleMatrix tanTerm = new SimpleMatrix(H,1);
		SimpleMatrix z = zFunction(xVector);
		for(int index = 0; index < H; index++){
			tanTerm.set(index,0,(1.0-Math.pow(Math.tanh(z.get(index,0)),2)));
		}
		WGrad = tanTerm.elementMult(U.transpose().mult(delta2)).mult(xVector.transpose());
	}

	private void wRegGradient(SimpleMatrix xVector, SimpleMatrix trueLabel){
		SimpleMatrix delta2 = pFunction(xVector).minus(trueLabel);
		SimpleMatrix tanTerm = new SimpleMatrix(H,1);
		SimpleMatrix z = zFunction(xVector);
		for(int index = 0; index < H; index++){
			tanTerm.set(index,0,(1.0-Math.pow(Math.tanh(z.get(index,0)),2)));
		}
		WGrad = (tanTerm.elementMult(U.transpose().mult(delta2)).mult(xVector.transpose())).plus(W.scale(lambda));
	}

	private void b1Gradient(SimpleMatrix xVector, SimpleMatrix trueLabel){
		SimpleMatrix delta2 = pFunction(xVector).minus(trueLabel);
		SimpleMatrix tanTerm = new SimpleMatrix(H,1);
		SimpleMatrix z = zFunction(xVector);
		for(int index = 0; index < H; index++){
			tanTerm.set(index,0,(1.0-Math.pow(Math.tanh(z.get(index,0)),2)));
		}
		b1Grad = (U.transpose().mult(delta2)).elementMult(tanTerm);
	}

	private void lGradient(SimpleMatrix xVector, SimpleMatrix trueLabel){
		SimpleMatrix delta2 = pFunction(xVector).minus(trueLabel);
		SimpleMatrix tanTerm = new SimpleMatrix(H,1);
		SimpleMatrix z = zFunction(xVector);
		for(int index = 0; index < H; index++){
			tanTerm.set(index,0,(1.0-Math.pow(Math.tanh(z.get(index,0)),2)));
		}
		LGrad = W.transpose().mult((U.transpose().mult(delta2)).elementMult(tanTerm));
	}


}