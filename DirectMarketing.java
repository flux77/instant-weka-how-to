/**
 * "Data mining in direct marketing (Simple)"
 *
 * Implement decision making to guide your marketing.
 * 
 *
 * @author http://bostjankaluza.net
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Random;


import weka.core.converters.ArffLoader;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class DirectMarketing {

	public static void main(String args[]) throws Exception {
		
		Instances trainData = new Instances(new BufferedReader(new FileReader("dataset/coil-train.arff")));
		
		trainData.setClassIndex(trainData.numAttributes() - 1);

		J48 j48 = new J48();
		j48.setOptions(new String[]{
			"-C", "0.25",	//set confidence factor
			"-M", "2"		//set min num of instances in leafes
		});
		double precision = crossValidation(j48, trainData);
		System.out.println(precision);
		
		NaiveBayes nb = new NaiveBayes();
		precision = crossValidation(nb, trainData);
		
		nb.buildClassifier(trainData);
		
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("dataset/coil-test.arff"));
		Instances testData = loader.getStructure();
		testData.setClassIndex(trainData.numAttributes() - 1);
		Instance current;
		while ((current = loader.getNextInstance(testData)) != null){
		   double cls = nb.classifyInstance(current);
		   System.out.println(cls);
		}
	
		

	}
	
	public static double crossValidation(Classifier cls, Instances data) throws Exception{
		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(cls, data, 10, new Random(1));
		System.out.println(eval.toSummaryString(false));
		System.out.println(eval.precision(1));
		return eval.precision(1);
	}

}
