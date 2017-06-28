/**
 * "Using Weka for stock value forecasting (Advanced)"
 *
 * Predict next day closing price. Note, this code requires 
 * 	- Weka 3-7 and 
 *  - Timeseries Forecasting 1.0 module.
 * Please check:
 * http://sourceforge.net/projects/weka/files/weka-packages/
 *
 * @author http://bostjankaluza.net
 */

import java.io.*;

import java.util.List;
import weka.core.Instances;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.WekaForecaster;

public class StockForecast {

	public static void main(String[] args) throws Exception {

			// load the stock data
			Instances dataset = new Instances(new BufferedReader(new FileReader("dataset/appl.arff")));
			dataset.sort(0);

			WekaForecaster forecaster = new WekaForecaster();

			// set the targets we want to forecast, that is, Close price
			forecaster.setFieldsToForecast("Close");

			forecaster.setBaseForecaster(new GaussianProcesses());

			forecaster.getTSLagMaker().setTimeStampField("Date"); // date time stamp
			forecaster.getTSLagMaker().setMinLag(12);
			forecaster.getTSLagMaker().setMaxLag(24); // monthly data

			// build the model
			forecaster.buildForecaster(dataset);
			forecaster.primeForecaster(dataset);

			List<List<NumericPrediction>> forecast = forecaster.forecast(1, System.out);

			// output the predictions
			List<NumericPrediction> predsAtStep = forecast.get(0);
			NumericPrediction predForTarget = predsAtStep.get(0);
			System.out.print("" + predForTarget.predicted() + " ");
			System.out.println();
	
	}
}