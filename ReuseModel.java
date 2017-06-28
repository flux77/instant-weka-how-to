/**
 * "Re-using models (Intermediate)"
 *
 * Save and load a model.
 * 
 * @author http://bostjankaluza.net
 */


import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.trees.J48;
import weka.core.Instances;

public class ReuseModel{

	public static void main(String args[]) throws Exception{
		
		J48 cls = new J48();
		 
		// train
		Instances inst = new Instances(new BufferedReader(
		                      new FileReader("dataset/titanic.arff")));
		inst.setClassIndex(inst.numAttributes() - 1);
		cls.buildClassifier(inst);
		weka.core.SerializationHelper.write("j48.model", cls);

		J48 cls2 = (J48) weka.core.SerializationHelper.read("j48.model");
		System.out.println(cls2);

		
	}

}
