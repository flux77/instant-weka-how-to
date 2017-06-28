/**
 * "Training a classifier (Simple)"
 *
 * Build and use a classifier.
 *
 * @author http://bostjankaluza.net
 */

import java.io.File;

import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Classifier {

	public static void main(String args[]) throws Exception{

		Instances dataset = LoadData.loadDataset("dataset/titanic.arff");
		dataset.setClassIndex(dataset.numAttributes()-1);

		// decision trees
		String[] options = new String[1];
		options[0] = "-U";
		J48 tree = new J48();
		tree.setOptions(options);
		tree.buildClassifier(dataset);	
		System.out.println(tree);

		// support vector machines
		SMO svm = new SMO();
		svm.buildClassifier(dataset);
		System.out.println(svm);

		//incremental Naive Bayes classifier
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("dataset/titanic.arff"));
		Instances dataStructure = loader.getStructure();
		dataStructure.setClassIndex(dataStructure.numAttributes() - 1);
		NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
		nb.buildClassifier(dataStructure);
		Instance current;
		while ((current = loader.getNextInstance(dataStructure)) != null)
			nb.updateClassifier(current);
		System.out.println(nb);
	}

}
