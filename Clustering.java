/**
 * "Clustering (Simple)"
 *
 * Build, evaluate, and use clusters. Note, the code below requires 
 * lots of processing, which may cause your computer to hang for a while.
 *
 * @author http://bostjankaluza.net
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Cobweb;
import weka.clusterers.EM;

public class Clustering {

	public static void main(String args[]) throws Exception{
		
		
		//load data
		Instances data = new Instances(new BufferedReader(new FileReader("dataset/bank-data.arff")));
		
		// new instance of clusterer
		EM model = new EM();
		// build the clusterer
		model.buildClusterer(data);
		System.out.println(model);
		
		clusterClassify();
		incrementalCluster();
		evaluate();

	}


	public static void clusterClassify() throws Exception{
		
		//load data
		Instances data = new Instances(new BufferedReader(new FileReader("dataset/bank-data.arff")));
		Instance inst = data.instance(0);
		data.delete(0);
		
		// new instance of clusterer
		EM model = new EM();
		// build the clusterer
		model.buildClusterer(data);
		//System.out.println(model);
		
		int cls = model.clusterInstance(inst);
		System.out.println("Cluster: "+cls);
		
		double[] dist = model.distributionForInstance(inst);
		for(int i = 0; i < dist.length; i++)	System.out.println("Cluster "+i+".\t"+dist[i]);

	}
	
	public static void incrementalCluster() throws Exception{
		// load data
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("dataset/bank-data.arff"));
		Instances data = loader.getStructure();
		 
		 // train Cobweb
		Cobweb model = new Cobweb();
		model.buildClusterer(data);
		Instance current;
		while ((current = loader.getNextInstance(data)) != null)
		   model.updateClusterer(current);
		model.updateFinished();
		System.out.println(model);

	}
	
	public static void evaluate() throws Exception{

		 Instances data = new Instances(new BufferedReader(new FileReader("dataset/bank-data.arff")));
		 
		 EM model = new EM();
		 
		 //double logLikelyhood = ClusterEvaluation.crossValidateModel(model, data, 10, new Random(1));
		 //System.out.println(logLikelyhood);
		 
		 ClusterEvaluation eval = new ClusterEvaluation();
		 model.buildClusterer(data);                                 // build clusterer
		 eval.setClusterer(model);                                   // the cluster to evaluate
		 eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
		 System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters


	}
}
