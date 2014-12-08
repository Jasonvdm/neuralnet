package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
    
    public static void main(String[] args) throws IOException {
	if (args.length < 3) {
	    System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev fileName");
	    return;
	}	    

	// this reads in the train and test datasets
	List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
	List<Datum> testData = FeatureFactory.readTestData(args[1]);
	String outPutFile = args[2];	
	
	//	read the train and test data
	//TODO: Implement this function (just reads in vocab and word vectors)
	FeatureFactory.initializeVocab("../data/vocab.txt");
	SimpleMatrix allVecs= FeatureFactory.readWordVectors("../data/wordVectors.txt");

	// initialize model 
	WindowModel model = new WindowModel(7, 100,0.001,0.001);
	System.out.println(outPutFile);
	HashMap<String, Integer> wordToNum = FeatureFactory.wordToNum;
	model.initWeights(allVecs, wordToNum);


	//TODO: Implement those two functions
	model.train(trainData);
	model.test(testData, outPutFile);
	model.baselineTrain(trainData);
	model.baselineTest(testData);
    }
}