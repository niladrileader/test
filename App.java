package com.niladri.weka.weka_app;

import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * Hello world!
 *
 */
public class App {
	public static double SVMpredict(Instances trainDataSet, Instances testDataSet) throws Exception {

		SMO svm = new SMO();
		String[] options=new String[8];
		options[0]="-C";
		options[1]="3";
		options[2]="-L";
		options[3]="0.13";
		options[4]="-P";
		options[5]="0.5";
		options[6]="-V";
		options[7]="1000";
		svm.setOptions(options);
		
		svm.buildClassifier(trainDataSet);

		System.out.println("===================");
		System.out.println("Actual Class, NB Predicted");
		double count = 0;
		for (int i = 0; i < testDataSet.numInstances(); i++) {
			double actualClass = testDataSet.instance(i).classValue();
			String actual = testDataSet.classAttribute().value((int) actualClass);
			Instance newInst = testDataSet.instance(i);
			double predNB = svm.classifyInstance(newInst);
			String predString = testDataSet.classAttribute().value((int) predNB);
			System.out.println(actual + ", " + predString);
			if (actualClass == predNB)
				count++;

		}

		double accuracy = count / testDataSet.numInstances();
		return accuracy;

	}
	public static double Treepredict(Instances trainDataSet, Instances testDataSet) throws Exception {

		J48 tree = new J48();
		tree.buildClassifier(trainDataSet);

		System.out.println("===================");
		System.out.println("Actual Class, NB Predicted");
		double count = 0;
		for (int i = 0; i < testDataSet.numInstances(); i++) {
			double actualClass = testDataSet.instance(i).classValue();
			String actual = testDataSet.classAttribute().value((int) actualClass);
			Instance newInst = testDataSet.instance(i);
			double predNB = tree.classifyInstance(newInst);
			String predString = testDataSet.classAttribute().value((int) predNB);
			System.out.println(actual + ", " + predString);
			if (actualClass == predNB)
				count++;

		}

		double accuracy = count / testDataSet.numInstances();
		return accuracy;

	}

	public static double NBpredict(Instances trainDataSet, Instances testDataSet) throws Exception {

		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(trainDataSet);

		System.out.println("===================");
		System.out.println("Actual Class, NB Predicted");
		double count = 0;
		for (int i = 0; i < testDataSet.numInstances(); i++) {
			double actualClass = testDataSet.instance(i).classValue();
			String actual = testDataSet.classAttribute().value((int) actualClass);
			Instance newInst = testDataSet.instance(i);
			System.out.println(newInst);
			double predNB = nb.classifyInstance(newInst);
			String predString = testDataSet.classAttribute().value((int) predNB);
			System.out.println(actual + ", " + predString);
			if (actualClass == predNB)
				count++;

		}

		double accuracy = count / testDataSet.numInstances();
		return accuracy;

	}
	
	public static String NBpredict(Instances trainDataSet, Instance newInst) throws Exception {

		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(trainDataSet);
		double count = 0;
			double predNB = nb.classifyInstance(newInst);
			String predString = newInst.classAttribute().value((int) predNB);
			

		

		
		return predString;

	}


	public static void main(String s[]) throws Exception {
		DataSource source = new DataSource("Training_files/iris.arff");
		Instances dataSet = source.getDataSet();
		dataSet.setClassIndex(dataSet.numAttributes() - 1);
		Instance aInstance=dataSet.firstInstance();
		System.out.println(aInstance);
		
		aInstance.setValue(aInstance.attribute(0), 5.8);
		aInstance.setValue(dataSet.attribute(1), 2.7);
		aInstance.setValue(dataSet.attribute(2), 5.1);
		aInstance.setValue(dataSet.attribute(3), 1.9);
		aInstance.setValue(dataSet.attribute(4), 1);
		
		
		
		//,,,,Iris-virginica
		
		System.out.println(aInstance);
		System.out.println(NBpredict(dataSet,aInstance));
		
//		dataSet.setClassIndex(5);
				/*  					CVParameterSelection ps = new CVParameterSelection();
									    ps.setClassifier(new SMO());
									    ps.setNumFolds(5);  // using 5-fold CV
									    ps.addCVParameter("C 1 3 10");
									    ps.addCVParameter("L 1.0E-5 0.2 10");
									    ps.addCVParameter("P 0.01 1 10");
									      
			
			//	      					build and output best options
				      					ps.buildClassifier(dataSet);
				      					System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
				      
				      
	      
	 
		int numClasses = dataSet.numClasses();
		for (int i = 0; i < numClasses; i++) {
			String classValue = dataSet.classAttribute().value(i);
			System.out.println("Class Value " + i + " is " + classValue);
		}

		double SVMavgAcc = 0, NBavgAcc = 0,TreeavgAcc = 0;
		;
		for (int i = 0; i < 100; i++) {
			dataSet.randomize(new Random());

			RemovePercentage rmvp1 = new RemovePercentage();
			rmvp1.setInputFormat(dataSet);
			rmvp1.setPercentage(70);
			rmvp1.setInvertSelection(true);
			Instances trainDataSet = Filter.useFilter(dataSet, rmvp1);

			RemovePercentage rmvp2 = new RemovePercentage();
			rmvp2.setInputFormat(dataSet);
			rmvp2.setPercentage(70);
			rmvp2.setInvertSelection(false);
			Instances testDataSet = Filter.useFilter(dataSet, rmvp2);
			System.out.println(
					dataSet.numInstances() + "  " + trainDataSet.numInstances() + "  " + testDataSet.numInstances());
			SVMavgAcc += SVMpredict(trainDataSet, testDataSet);
//			NBavgAcc += NBpredict(trainDataSet, testDataSet);
//			TreeavgAcc+=Treepredict(trainDataSet, testDataSet);
		}

		System.out.print("Avg Accuracy for SVM:" + SVMavgAcc+"\t");
//		System.out.print("Avg Accuracy for NB:" + NBavgAcc+"\t");
//		System.out.print("Avg Accuracy for J48:" + TreeavgAcc+"\t");
	*/
	}
	

}