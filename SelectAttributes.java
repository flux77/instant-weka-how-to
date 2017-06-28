/**
 * "Selecting attributes (Intermediate)"
 *
 * Select relevant attributes, apply principal component analysis.
 *
 * @author http://bostjankaluza.net
 */

import java.util.Random;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
//import weka.attributeSelection.AttributeSelection

public class SelectAttributes{
	
	public static void main(String args[]) throws Exception{
		
		selectExample();
		PCAExample();
		ClassifierSpecificExample();
		
	}
	
	public static void selectExample() throws Exception{
		
		Instances data = LoadData.loadDataset("dataset/titanic.arff");
		
		AttributeSelection filter = new AttributeSelection();  // package weka.filters.supervised.attribute!
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(data);
		// generate new data
		Instances newData = Filter.useFilter(data, filter);
		System.out.println(newData.numAttributes());

	}

	public static void infoGainExample() throws Exception{
		
		Instances data = LoadData.loadDataset("dataset/titanic.arff");
		
		weka.attributeSelection.AttributeSelection attSelect = new weka.attributeSelection.AttributeSelection();  // package weka.attributeSelection.AttributeSelection!
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker search = new Ranker();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);
		attSelect.SelectAttributes(data);
		int[] indices = attSelect.selectedAttributes();
		System.out.println(attSelect.toResultsString());
		System.out.println(Utils.arrayToString(indices));

	}
	


	public static void PCAExample() throws Exception{
		
		Instances data = LoadData.loadDataset("dataset/titanic.arff");
		
		AttributeSelection filter = new AttributeSelection();  // package weka.filters.supervised.attribute!
		PrincipalComponents eval = new PrincipalComponents();
		Ranker search = new Ranker();
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(data);
		// generate new data
		Instances newData = Filter.useFilter(data, filter);
		System.out.println(newData);

	}
	
	public static void ClassifierSpecificExample() throws Exception{
		Instances data = LoadData.loadDataset("dataset/titanic.arff");
		data.setClassIndex(data.numAttributes()-1);
		
		
		AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
		ReliefFAttributeEval eval = new ReliefFAttributeEval ();
		Ranker search = new Ranker();
		J48 baseClassifier = new J48();
		classifier.setClassifier(baseClassifier);
		classifier.setEvaluator(eval);
		classifier.setSearch(search);
		Evaluation evaluation = new Evaluation(data);
		evaluation.crossValidateModel(classifier, data, 10, new Random(1));
		System.out.println(evaluation.toSummaryString());


	}
	
	
}