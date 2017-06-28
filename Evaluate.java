/**
 * "Testing and evaluating your models (Simple)"
 *
 * Test and estimate model performance.
 * 
 * @author http://bostjankaluza.net
 */

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import java.util.Random;

import javax.swing.*;

import weka.core.*;
import weka.classifiers.evaluation.*;
import weka.gui.visualize.*;

public class Evaluate {

	public static void main(String args[]) throws Exception {

		evaluate("dataset/titanic.arff");
		ROCCurve("dataset/titanic.arff");
	}

	public static void evaluate(String dataset) throws Exception {

		DataSource source = new DataSource(dataset);

		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		J48 classifier = new J48();

		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(classifier, data, 10, new Random(1));

		System.out.println(eval.toSummaryString("Results", false));

		System.out.println("----------------");
		System.out.println(eval.correct());
		System.out.println(eval.pctCorrect());
		System.out.println(eval.kappa());
		System.out.println(eval.correct());

		System.out.println("----------------");
		System.out.println(eval.toMatrixString());

	}

	public static void ROCCurve(String dataset) throws Exception {

		// load data
		DataSource source = new DataSource(dataset);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);

		// train classifier
		J48 cl = new J48();
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(cl, data, 10, new Random(1));

		// generate curve
		ThresholdCurve tc = new ThresholdCurve();
		int classIndex = 0;
		Instances result = tc.getCurve(eval.predictions(), classIndex);

		// plot curve
		ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
		vmc.setROCString("(Area under ROC = "+ Utils.doubleToString(tc.getROCArea(result), 4) + ")");
		vmc.setName(result.relationName());
		PlotData2D tempd = new PlotData2D(result);
		tempd.setPlotName(result.relationName());
		tempd.addInstanceNumberAttribute();
		// specify which points are connected
		boolean[] cp = new boolean[result.numInstances()];
		for (int n = 1; n < cp.length; n++)
			cp[n] = true;
		tempd.setConnectPoints(cp);
		// add plot
		vmc.addPlot(tempd);

		// display curve		
		JFrame frame = new javax.swing.JFrame("ROC Curve");
		frame.setSize(800, 500);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(vmc);
		frame.setVisible(true);

	}

}