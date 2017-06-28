/**
 * "Associations rules (Simple)"
 *
 * Find requent patterns.
 *
 * @author http://bostjankaluza.net
 */

import java.io.BufferedReader;
import java.io.FileReader;

import weka.core.Instances;
import weka.associations.Apriori;

public class AssociationRules{

	public static void main(String args[]) throws Exception{
		
		//load data
		Instances data = new Instances(new BufferedReader(new FileReader("dataset/bank-data.arff")));
		
		//build model
		Apriori model = new Apriori();
		model.buildAssociations(data); 
		System.out.println(model);
		
	}

}
