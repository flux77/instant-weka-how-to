/**
 * "Filtering attributes (Simple)"
 *
 * Apply filter to remove attributes.
 *
 * @author http://bostjankaluza.net
 */

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class FilterAttributes{
	
	public static void main(String args[]) throws Exception{
		
		filterExample();
		
	}
	
	public static void filterExample() throws Exception{
		Instances data = LoadData.loadDataset("dataset/titanic.arff");
		
		String[] options = new String[2];
		options[0] = "-R";                                    // "range"
		options[1] = "2";                                     // first attribute
		Remove remove = new Remove();                         // new instance of filter
		remove.setOptions(options);                           // set options
		remove.setInputFormat(data);                          // inform filter about dataset **AFTER** setting options
		Instances newData = Filter.useFilter(data, remove);   // apply filter

		System.out.println(data.toString());
		System.out.println(newData.toString());
		
	}
	
	
	
}