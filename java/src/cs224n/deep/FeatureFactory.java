package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;
import org.ejml.simple.SimpleMatrix;


public class FeatureFactory {


	private FeatureFactory() {

	}

	 
	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData= read(filename);
        return trainData;
	}
	
	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData= read(filename);
        return testData;
	}
	
	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
	    // TODO: you'd want to handle sentence boundaries
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}

		return data;
	}
 
 
	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	//access it directly in WindowModel
	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		SimpleMatrix allVecs = new SimpleMatrix(100232, 50); 
		int wordIndex = 0;
		try{
			BufferedReader br = new BufferedReader(new FileReader(vecFilename));
			String line;
			while ((line = br.readLine()) != null) {
				String[] stringVec = line.split(" ");
				int index = 0;
				for (String s : stringVec) {
					double value = Double.parseDouble(s);
					allVecs.set(wordIndex, index, value);
					index++;
				}
			    wordIndex++;
			}
			br.close();
		}
		catch(Exception e){
			e.printStackTrace();
		}
		if (allVecs!=null) return allVecs;
		return null;
	}

	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {
		int wordIndex = 0;
		try{
			BufferedReader br = new BufferedReader(new FileReader(vocabFilename));
			String line;
			while ((line = br.readLine()) != null) {
				wordToNum.put(line, wordIndex);
				numToWord.put(wordIndex, line);
				wordIndex++;
			}
			br.close();
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return wordToNum;
	}
 








}
