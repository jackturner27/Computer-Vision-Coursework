#define _CRT_SECURE_NO_DEPRECATE
#include <time.h>  
#include <math.h>  
#include <ctype.h>  
#include <stdio.h>  
#include <iostream>
#include <stdlib.h>
#include <opencv/cv.h>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/highgui.h> 
#include <opencv/ml.h>	
#include <fstream>
#include <sstream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

// --- Flann Based Matcher ---
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

// --- Brute Force Matcher ---
//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");

// --- SURF Detector and Extractor ---
Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();
SurfFeatureDetector detector(100);

// --- SIFT Detector and Extractor ---
//Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor();
//SiftFeatureDetector detector(100);

int clusterSize = 10;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 3;
int flags = KMEANS_PP_CENTERS;
BOWKMeansTrainer bowTrainer(clusterSize, tc, retries, flags);
BOWImgDescriptorExtractor bowDE(extractor, matcher);

int Motion_num = 0;
int Motion_flag = 0;
const double MHI_DURATION = 1;
const double threshold_value = 25;
const String PATH = ""; //use this if you want a specific path to the datasets

int results = 0;
Mat training_data(0, clusterSize, CV_32FC1);
Mat labels(0, 1, CV_32SC1);

int count_lines(char *filename)
{
	ifstream file;
	int n = 0;
	string temp;
	file.open(filename, ios::in);
	if (file.fail()) {
		return 0;
	}
	else {
		while (getline(file, temp)) {
			n++;
		}
		return n;
	}
	file.close();
}

Mat image_binary_prev_5, image_binary, image_binary_diff_5;
Mat SAMHI_10 = Mat(480, 640, CV_32FC1, Scalar(0, 0, 0));

Mat generate_MHI(string fname, int frame_diff) {

	VideoCapture cap;
	cap = VideoCapture(fname);

	Mat image_binary_prev_5, image_binary, image_binary_diff_5;
	int nframes = cap.get(CV_CAP_PROP_FRAME_COUNT);
	for (int i = 1; i < nframes; i++)
	{
		Mat frame;
		cap.read(frame);
		cvtColor(frame, image_binary, CV_BGR2GRAY);

		if (i == 1) {
			image_binary_prev_5 = image_binary.clone();
		}

		if (i % frame_diff == 0) {
			absdiff(image_binary_prev_5, image_binary, image_binary_diff_5);
			image_binary_prev_5 = image_binary.clone();
		}

		if (i == frame_diff + 1) {
			threshold(image_binary_diff_5, image_binary_diff_5, threshold_value, 255, THRESH_BINARY);
			Size framesize = image_binary_diff_5.size();
			int h = framesize.height;
			int w = framesize.width;
			SAMHI_10 = Mat(h, w, CV_32FC1, Scalar(0, 0, 0));
			updateMotionHistory(image_binary_diff_5, SAMHI_10, (double)i / nframes, MHI_DURATION);
		}
		if (i > frame_diff + 1 && i % frame_diff == 0) {
			threshold(image_binary_diff_5, image_binary_diff_5, threshold_value, 255, THRESH_BINARY);
			updateMotionHistory(image_binary_diff_5, SAMHI_10, (double)i / nframes, MHI_DURATION);
		}
	}

	return SAMHI_10;
}

int generate_vocab() {

	fstream in("train_2.txt");
	int num_lines = count_lines("train_2.txt");

	for (int fnum = 1; fnum <= num_lines; fnum++)
	{
		string fname;
		string filename;
		int lab;
		in >> lab;
		in >> filename;
		fname = PATH + filename;

		SAMHI_10 = generate_MHI(fname, 5);
		
		cout << "File " << fname << " MHI done.." << endl;

		vector<KeyPoint> keypoint;
		SAMHI_10.convertTo(SAMHI_10, CV_8UC1, 255, 0);
		detector.detect(SAMHI_10, keypoint);
		Mat features;
		Mat keyP1, keyP2;
		Mat allDescriptors;
		extractor->compute(SAMHI_10, keypoint, features);
		allDescriptors.push_back(features);
		bowTrainer.add(features);
	}

	in.close();
	FileStorage fs("Vocabulary.xml", FileStorage::WRITE);
	Mat dictionary = bowTrainer.cluster();

	fs << "vocabulary" << dictionary;
	fs.release();
	cout << "Generated Vocabulary" << endl;

	return 0;
}

int train_data() {

	fstream in("train_2.txt");

	FileStorage fs;
	Mat dict;
	fs.open("Vocabulary.xml", FileStorage::READ);
	fs["vocabulary"] >> dict;
	bowDE.setVocabulary(dict);

	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;
	int num_lines = count_lines("train_2.txt");
	for (int fnum = 1; fnum <= num_lines; fnum++)
	{
		string fname;
		int l;
		in >> l;
		in >> fname;
		string filename;
		filename = PATH + fname;

		SAMHI_10 = generate_MHI(filename, 5);
		SAMHI_10.convertTo(SAMHI_10, CV_8UC1, 255, 0);
		detector.detect(SAMHI_10, keypoint1);
		bowDE.compute(SAMHI_10, keypoint1, bowDescriptor1);
		labels.push_back(l);
		training_data.push_back(bowDescriptor1);
	}
	in.close();

	return 0;
}

int NormalBayes_train() {

	train_data();

	CvNormalBayesClassifier nb_classifier;
	nb_classifier.train(training_data, labels);
	nb_classifier.save("nb_data.xml");

	return 0;
}

int SVM_train() {

	train_data();

	CvSVMParams params;
	params.kernel_type = CvSVM::RBF;
	params.svm_type = CvSVM::C_SVC;
	params.gamma = 0.50625000000000009;
	params.C = 312.50000000000000;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);
	CvSVM svm_classifier;
	svm_classifier.train_auto(training_data, labels, cv::Mat(), cv::Mat(), params);
	svm_classifier.save("svm_data.xml");
	CvSVMParams params_re = svm_classifier.get_params();
	printf("\nParms: C = %f,gamma = %f \n", params_re.C, params_re.gamma);
	
	return 0;
}

int NormalBayes_test() {

	fstream in("test_2.txt");
	int LineNumbers;
	LineNumbers = count_lines("test_2.txt");

	Mat dictionary;
	FileStorage fs2("Vocabulary.xml", FileStorage::READ);

	if (fs2.isOpened())
	{
		fs2["vocabulary"] >> dictionary;
	}
	bowDE.setVocabulary(dictionary);

	CvNormalBayesClassifier nb_classifier;
	nb_classifier.load("nb_data.xml");

	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, clusterSize, CV_32FC1);
	vector<KeyPoint> keypoint2;
	Mat bowDescriptor2;

	Mat results(0, 1, CV_32FC1);

	for (int fnum = 1; fnum <= LineNumbers; fnum++)
	{
		string fname;
		string path1;
		string path2;
		string filename;
		float lab;
		in >> lab;
		in >> filename;
		string filename1 = PATH + filename;

		SAMHI_10 = generate_MHI(filename1, 5);

		vector<KeyPoint> keypoint;
		SAMHI_10.convertTo(SAMHI_10, CV_8UC1, 255, 0);
		detector.detect(SAMHI_10, keypoint);
		bowDE.compute(SAMHI_10, keypoint, bowDescriptor2);
		evalData.push_back(bowDescriptor2);
		groundTruth.push_back(lab);

		float result = nb_classifier.predict(bowDescriptor2);
		cout << "File " << fnum << " tested:: " << result << endl;
		results.push_back(result);
	}

	FileStorage tra1("test.xml", FileStorage::WRITE);
	if (!tra1.isOpened())
	{
		cerr << "failed to open " << endl;
	}
	tra1 << "trainData" << evalData << "label" << groundTruth;
	tra1.release();

	double errorRate = (double)countNonZero(groundTruth - results) / evalData.rows;
	double accuracy = (double)1 - errorRate;
	printf("%s%f%s", "Accuracy is ", accuracy * 100, "%\n");
	return 0;
}

int SVM_test() {

	fstream in("test_2.txt");
	int num_lines = count_lines("test_2.txt");

	Mat dictionary;
	FileStorage fs2("Vocabulary.xml", FileStorage::READ);

	if (fs2.isOpened())
	{
		fs2["vocabulary"] >> dictionary;
	}
	bowDE.setVocabulary(dictionary);

	CvSVM svm_classifier;
	svm_classifier.load("svm_data.xml");

	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, clusterSize, CV_32FC1);
	vector<KeyPoint> keypoint2;
	Mat bowDescriptor2;

	Mat results(0, 1, CV_32FC1);

	for (int fnum = 1; fnum <= num_lines; fnum++)
	{
		string fname;
		string path1;
		string path2;
		string filename;
		float lab;
		in >> lab;
		in >> filename;
		string filename1 = PATH + filename;

		SAMHI_10 = generate_MHI(filename1, 5);

		vector<KeyPoint> keypoint;
		SAMHI_10.convertTo(SAMHI_10, CV_8UC1, 255, 0);
		detector.detect(SAMHI_10, keypoint);
		bowDE.compute(SAMHI_10, keypoint, bowDescriptor2);
		evalData.push_back(bowDescriptor2);
		groundTruth.push_back(lab);

		float result = svm_classifier.predict(bowDescriptor2);
		cout << "File " << fnum << " tested:: " << result << endl;
		results.push_back(result);
	}

	FileStorage tra1("test.xml", FileStorage::WRITE);
	if (!tra1.isOpened())
	{
		cerr << "failed to open " << endl;
	}
	tra1 << "trainData" << evalData << "label" << groundTruth;
	tra1.release();

	double errorRate = (double)countNonZero(groundTruth - results) / evalData.rows;
	double accuracy = (double)1 - errorRate;
	printf("%s%f%s", "Accuracy is ", accuracy * 100, "%\n");
	return 0;
}

int svm_generate_confusion_matrix() {

	cv::FileStorage fs("data.xml", cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		cerr << "failed to open " << endl;
	}
	Mat trainingData;
	fs["trainData"] >> trainingData;
	fs["label"] >> labels;
	int filesize = trainingData.rows;
	int folders = 10;
	int validateVol = (int)(filesize / folders);
	int labelline = labels.rows;

	srand(time(0));

	Mat conmat = Mat::zeros(10, 10, CV_32SC1);
	for (int i = 0; i<folders; i++)
	{

		Mat trainS(filesize - validateVol, clusterSize, CV_32FC1);
		Mat validS(validateVol, clusterSize, CV_32FC1);
		Mat Tlabel(filesize - validateVol, 1, CV_32SC1);
		Mat Vlabel(validateVol, 1, CV_32SC1);
		int *markIdx = new int[filesize];
		for (int j = 0; j<filesize; j++)
		{
			markIdx[j] = 0;
		}
		for (int j = 0; j<validateVol; j++)
		{
			int a = rand() % filesize;
			while (markIdx[a] == 1)
			{
				a = rand() % filesize;
			}
			markIdx[a] = 1;
		}
		int countT = 0;
		int countV = 0;
		for (int j = 0; j<filesize; j++)
		{
			if (markIdx[j] == 0)
			{
				Mat trainS_row = trainS.row(countT);
				trainingData.row(j).copyTo(trainS_row);
				Tlabel.at<int>(countT) = labels.at<int>(j);

				/*Mat Tlabel_row = Tlabel.row(countT);
				labels.row(j).copyTo(Tlabel_row);*/
				int a = labels.at<int>(j);
				countT++;
			}
			else
			{
				Mat validS_row = validS.row(countV);
				trainingData.row(j).copyTo(validS_row);

				/*	Mat Vlabel_row = Vlabel.row(countV);
				labels.row(j).copyTo(Vlabel_row);*/
				Vlabel.at<int>(countV) = labels.at<int>(j);
				countV++;
			}
		}

		CvSVM SVM;
		CvSVMParams paras;
		paras.svm_type = CvSVM::C_SVC;
		paras.kernel_type = CvSVM::RBF;
		paras.gamma = 0.50625000000000009;
		paras.C = 312.50000000000000;
		paras.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.000001);
		bool res = SVM.train_auto(trainS, Tlabel, Mat(), Mat(), paras);
		CvSVMParams params_re = SVM.get_params();
		printf("\nParms: C = %f,gamma = %f \n", params_re.C, params_re.gamma);

		for (int j = 0; j<validateVol; j++)
		{

			int motion = SVM.predict(validS.row(j));
			cout << Vlabel.at<int>(j) << ' ' << (int)motion << endl;
			conmat.at<int>(Vlabel.at<int>(j) - 1, (int)(motion)-1)++;
			
			//double err = (double)countNonZero(Vlabel.at<float>(j) - motion) / validS.rows;
			//printf("%s%f", "Accuracy : ", (1-err)*100);
			//printf("%\n");
		}
	}
	cout << endl << conmat << endl;
	return 0;
}

int nb_generate_confusion_matrix() {

	cv::FileStorage fs("data.xml", cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		cerr << "failed to open " << endl;
	}
	Mat trainingData;
	fs["trainData"] >> trainingData;
	fs["label"] >> labels;
	int filesize = trainingData.rows;
	int folders = 10;
	int validateVol = (int)(filesize / folders);
	int labelline = labels.rows;

	srand(time(0));

	Mat conmat = Mat::zeros(10, 10, CV_32SC1);
	for (int i = 0; i<folders; i++)
	{

		Mat trainS(filesize - validateVol, clusterSize, CV_32FC1);
		Mat validS(validateVol, clusterSize, CV_32FC1);
		Mat Tlabel(filesize - validateVol, 1, CV_32SC1);
		Mat Vlabel(validateVol, 1, CV_32SC1);
		int *markIdx = new int[filesize];
		for (int j = 0; j<filesize; j++)
		{
			markIdx[j] = 0;
		}
		for (int j = 0; j<validateVol; j++)
		{
			int a = rand() % filesize;
			while (markIdx[a] == 1)
			{
				a = rand() % filesize;
			}
			markIdx[a] = 1;
		}
		int countT = 0;
		int countV = 0;
		for (int j = 0; j<filesize; j++)
		{
			if (markIdx[j] == 0)
			{
				Mat trainS_row = trainS.row(countT);
				trainingData.row(j).copyTo(trainS_row);
				Tlabel.at<int>(countT) = labels.at<int>(j);
				int a = labels.at<int>(j);
				countT++;
			}
			else
			{
				Mat validS_row = validS.row(countV);
				trainingData.row(j).copyTo(validS_row);
				Vlabel.at<int>(countV) = labels.at<int>(j);
				countV++;
			}
		}

		NormalBayesClassifier nb_classifier;
		nb_classifier.train(trainS, Tlabel);

		for (int j = 0; j<validateVol; j++)
		{

			int motion = nb_classifier.predict(validS.row(j));
			cout << Vlabel.at<int>(j) << ' ' << (int)motion << endl;
			conmat.at<int>(Vlabel.at<int>(j) - 1, (int)(motion)-1)++;

			double err = (double)countNonZero(Vlabel.at<float>(j) - motion) / validS.rows;
			printf("%s%f", "Accuracy : ", (1 - err) * 100);
			printf("%\n");
		}
	}
	cout << endl << conmat << endl;
	return 0;

}

//Useful for me to change this rather than comment/uncomment
//0 = SVM
//1 = Normal Bayes
int classifier = 1;

int main(int argc, char** argv) {

	auto start = high_resolution_clock::now();

	cout << "Training Dataset..." << endl;
	generate_vocab();

	switch (classifier) {
		case 0: //Use SVM
			cout << "Classifying with SVM..." << endl;
			SVM_train();
			SVM_test();
			break;
		case 1: //Use Normal Bayes
			cout << "Classifying with Normal Bayes..." << endl;
			NormalBayes_train();
			NormalBayes_test();
			break;
	}

	//This process is perhaps unnecessary since training_data can be read directly but it helps in debugging and testing
	cout << "Writing Data to XML" << endl;
	Mat data = training_data;
	FileStorage datatest("data.xml", FileStorage::WRITE);
	if (!datatest.isOpened())
	{
		cerr << "failed to open " << endl;
	}
	datatest << "trainData" << training_data << "label" << labels;
	datatest.release();

	switch (classifier) {
		case 0: //Use SVM
			cout << "Generating Confusion Matrix Using SVM..." << endl;
			svm_generate_confusion_matrix();
			break;
		case 1: //Use Normal Bayes
			cout << "Generating Confusion Matrix Using Normal Bayes..." << endl;
			nb_generate_confusion_matrix();
			break;
	}

	//Measure the execution time in seconds since this is a loooooong process
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(stop - start);
	cout << duration.count() << endl;

	return 0;
}