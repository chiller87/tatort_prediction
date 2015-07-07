




/*

	This is a little program, to convert and predict tatort DB, given as csv file, in various formats and with different algorithms.


	Filenames, related to data files, schould be given without file extension. Filenames that are used to write data to, should be given with file extension.
	The reason for this is, that the data files are named with a combination of different parts of the name,
	e.g. "tatortdb_clean_1_train.csv" is a combination of:
		dbCleanPrefix = "tatortdb_clean"
		an ID = "_1"
		trainSuffix = "_train"
	The correct file extension (DB_FILE_EXTENSION or LIBFM_FILE_EXTENSION) is added where it is needed.
	The filenames used to print some stuff are not build this way, and so they are defined where they are needed.


	Author: Simon Schiller

*/


#include <string>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <limits>
#include <thread>
#include <mutex>


#include "MyException.h"
#include "Logger.h"
#include "TatortTendencyParser.h"
#include "TatortTendencyPredictor.h"
#include "TatortFMParser.h"
#include "TatortFMPredictor.h"
#include "StringTokenizer.h"

using namespace std;



// LIBFM data as User-Episode matrix
#define DATA_UE_MATRIX 1
// LIBFM data as User-Episode-Detective tensor
#define DATA_UED_TENSOR 2
#define DATA_UED_TENSOR_PLUS_ATTRIBUTES 3


// file extension of DB files
#define DB_FILE_EXTENSION ".db"
// file extension of libfm files
#define LIBFM_FILE_EXTENSION ".libfm"




// struct of data belonging to each thres (parallel parameter testing)
typedef struct {
	int threadID;
	int start, stop, step;
} ThreadData_t;


// mutexe (?) used to avoid problems writing to same files in different threads
mutex sgdFileMutex_g;
mutex alsFileMutex_g;
mutex mcmcFileMutex_g;


// global definitions of filename parts
string dbPrefix_g = "tatortdb";
string dbCleanPrefix_g = "tatortdb_clean";
string trainSuffix_g = "_train";
string testSuffix_g = "_test";
string predSuffix_g = "_pred";
string targetSuffix_g = "_target";
string delimiter_g = "|";


// global definitions used to determine, what action to be performed
bool help_g = false;
bool parameterSearch_g = false;
bool runScenario_g = false;
bool nlfSearch_g = false;


// global definitions used for parameter search
int iterStart_g = 100, iterStop_g = 500, iterStep_g = 100;
double stdevStart_g = 0.0, stdevStop_g = 2.0, stdevStep_g = 0.5;
double regStart_g = 0.0, regStop_g = 1.0, regStep_g = 0.2;
double lrStart_g = 0.0001, lrStop_g = 0.001, lrStep_g = 0.0001;
unsigned int numOfThreads_g = 4;
bool mcmc_g = false, sgd_g = false, als_g = false;
bool includeNLF_g = false;
int nlf_g = 8;
int nlfStart_g = 8, nlfStop_g = 8, nlfStep_g = 2;
TatortFMPredictor nlfPredictor_g;


// global definitions used for scenario testing
double initStdev_g = 1.2, initLr_g = 0.05;
int initIter_g = 580;
string initAlgo_g = "mcmc", initReg_g = "1.0";
unsigned int numOfScenarios_g = 10;
vector<unsigned int> attributeIndicesToUse_g;
TatortFMPredictor matrixPredictor_g;
TatortFMPredictor tensorPredictor_g;
TatortFMPredictor tensorAttributesPredictor_g;
string matPredFile_g = "param_matrix.dat";
string tenPredFile_g = "param_tensor.dat";
string tenAttPredFile_g = "param_tensorAttributes.dat";


// general global definitions
int dataRepresentation_g = DATA_UE_MATRIX;






// modifying DB data
// deprecated
void addUserIdAndDetectiveId(string filename);
void addIdColumn(string filename, unsigned int columnId, string newColumnTitel);
void parseMapsAndWriteToFile(string filename);
void cleanDataset(string filename, int userRatingThreshold);
void divideDataIntoTrainAndTestData(string sourceFilename, int count, int trainPercentage);
void completeViewersAndQuotes(string filename, int viewerIndex, int quoteIndex);


// file operations
void writeToFile(string filename, string text);
void appendLineToFile(string filename, string text);
bool fileExist(string filename);
void readPredictorsFromFile();
TatortFMPredictor readPredictorFromFile(string filename);


// to string operations
string resultsToStringHuman(map<string, vector<double> >* predictionResults, vector<string>* methods, vector<double>* means);
string resultsToString(map<string, vector<double> >* predictionResults, vector<string>* methods, vector<double>* means);
string predictionToString(TatortFMPredictor *bestFmPredictor, double bestResult);
string predictorToString(TatortFMPredictor fmPred);


// parsing
void runFMParser(string trainFilename, string testFilename, int dataRepresentation);
void runFMParser(string trainFilename, string testFilename, int dataRepresentation, string outTrainFilename, string outTestFilename);


// predicting
double runFMPredictor(string trainFilename, string testFilename, string predFilename, TatortFMPredictor fmPredictor);
double tendencyTrainAndTest(string trainFilename, string testFilename, string predFilename);
double fmTrainAndTest(string trainFilename, string testFilename, string predFilename, TatortFMPredictor fmPredictor, int dataRepresentation);
void testScenario(const string *scenarioName, vector<double>* prediction);


// searching for best parameters
void searchingOptimalParams(string scenario, unsigned int numOfThreads = 1);
void fmParallelParameterChoicePerIteration(string trainFilename, string testFilename, ThreadData_t *td);
void fmParallelParameterChoicePerNLF(string trainFilename, string testFilename, ThreadData_t *td);
void checkParams(string trainFilename, string testFilename, string predFilename, TatortFMPredictor *currFmPredictor, TatortFMPredictor *bestFmPredictor, double *bestResult, string outFilename, mutex *fileMutex = NULL);
// deprecated
void fmNLFChoice(string scenario);
// deprecated
//void fmParallelParameterChoicePerAlgorithm(string trainFilename, string testFilename, string algorithm, TatortFMPredictor *bestFmPredictor);
// deprecated
//TatortFMPredictor fmSerialParameterChoice(string trainFilename, string testFilename);


// statistics
vector<double> computeMeanForResults(map<string, vector<double> >* predictionResults);

// output
void showHelp();

// generel
int parseCmdLineArguments(int argc, char **argv);












void showHelp() {

	int columnWidthOption = 16;
	int columnWidthParam = 8;

	cout << "=============================================================================" << endl;
	cout << "                                       HELP                                  " << endl;
	cout << "=============================================================================" << endl;
	cout << "param types:\tn = int, d = double, str = string" << endl;
	cout << setw(columnWidthOption) << "Option" << setw(columnWidthParam) << "Param" << "\tDescription" << endl;
	cout << "-----------------------------------------------------------------------------" << endl;
	cout << endl << "options used for testing scenarios:" << endl;
	cout << setw(columnWidthOption) << "-test" << setw(columnWidthParam) << "n"<< "\trun test for 'n' different scenarios" << endl;
	//cout << setw(columnWidthOption) << "-attr" << setw(columnWidthParam) << "n,n,..." << "\twhat attributes should be used in tensor+attributes scenario (indices of columns)" << endl;

	cout << endl << "options used for parameter search:" << endl;
	cout << setw(columnWidthOption) << "-search" << setw(columnWidthParam) << "n" << "\trun parameter search including (= 1) number of latent factors (nlf) or not (= 0)" << endl;
	cout << setw(columnWidthOption) << "-threads" << setw(columnWidthParam) << "n" << "\trun parameter search with number of threads (default is 1)" << endl;
	cout << setw(columnWidthOption) << "-algos" << setw(columnWidthParam) << "n,n,n" << "\trun parameter search for algorithm 'mcmc,als,sgd' (e.g. '1,0,1' for mcmc and sgd)" << endl;
	cout << setw(columnWidthOption) << "-iter" << setw(columnWidthParam) << "n,n,n" << "\trun parameter search with range of iterations 'start,stop,step'" << endl;
	cout << setw(columnWidthOption) << "-stdev" << setw(columnWidthParam) << "d,d,d" << "\trun parameter search with range of standard deviation 'start,stop,step'" << endl;
	cout << setw(columnWidthOption) << "-reg" << setw(columnWidthParam) << "d,d,d" << "\trun parameter search with range of regulation 'start,stop,step'" << endl;
	cout << setw(columnWidthOption) << "-lr" << setw(columnWidthParam) << "d,d,d" << "\trun parameter search with range of learning rates 'start,stop,step'" << endl;
	cout << setw(columnWidthOption) << "-k" << setw(columnWidthParam) << "n,n,n" << "\trun search with nlf range 'start, stop, step'" << endl;
	cout << setw(columnWidthOption) << "-dr" << setw(columnWidthParam) << "n" << "\trun parameter search with given representation of data (1=UIM, 2=UIT, 3=UIT+) (default = 2)" << endl;




	cout << endl << "general options:" << endl;
	cout << setw(columnWidthOption) << "-help, -h" << setw(columnWidthParam) << " " << "\tshow this screen" << endl;
	cout << endl << "if multiple options (-h, -ns, -psearch) are present, only one executed:" << endl;
	cout << "1. -h" << endl;
	cout << "2. -psearch" << endl;
	cout << "3. -ns" << endl;

	cout << endl << "Additional Information:" << endl << "==================================" << endl;
	cout << "to run scenarios there have to be 3 files present ('param_matrix.dat', 'param_tensor.dat', 'param_tensorAttributes.dat'), each representing the parameters for the prediction with the specific data representation:" << endl;
	cout << "The files should contain the following information with the given datatypes. Each line should contain one value in the following order:" << endl;
	cout << setw(9) << "(str)\t" << setw(15) << "algorithm" << "\t(schould be one of 'mcmc', 'als', or 'sgd')" << endl;
	cout << setw(9) << "(n)\t" << setw(15) << "iterations" << "\t(defines how many iterations should be used for training)" << endl;
	cout << setw(9) << "(d)\t" << setw(15) << "stdev" << "\t(defines the standard deviation for initializing)" << endl;
	cout << setw(9) << "(d)\t" << setw(15) << "reg" << "\t(defines the regulation used for training)" << endl;
	cout << setw(9) << "(d)\t" << setw(15) << "lr" << "\t(defines the learning rate used for training with SGD)" << endl;
	cout << setw(9) << "(n)\t" << setw(15) << "w0" << "\t(defines whether to use w0 (!= 0) or not (= 0))" << endl;
	cout << setw(9) << "(n)\t" << setw(15) << "w" << "\t(defines whether to use w (!= 0) or not (= 0))" << endl;
	cout << setw(9) << "(n)\t" << setw(15) << "k" << "\t(defines the number of latent factors to use)" << endl;

	cout << endl << "run param search without NLF will compute best params per iteration with NLF = 8, otherwise per NLF" << endl;


	cout << endl;
}






int main(int argc, char **argv) {





	//cout << "max random number = " << RAND_MAX << endl;
	clock_t begin = clock();
	// init logger
	Logger::getInstance()->setVerbosityLevel(LOG_DEBUG);

	srand(time(NULL));

	// initialize attributes to use for tensor-plus-attribute scenario (will be cleared, if cmd line option '-attr' is present)
	attributeIndicesToUse_g.push_back(2);
	attributeIndicesToUse_g.push_back(8);
	attributeIndicesToUse_g.push_back(9);

	parseCmdLineArguments(argc, argv);

	// show help menu
	if(help_g) {
		showHelp();
	}
	// searching for the best parameters
	else if(parameterSearch_g) {

		string scenario = dbCleanPrefix_g;
		try {
			searchingOptimalParams(scenario, numOfThreads_g);
		} catch(MyException e) {
			Logger::getInstance()->log(e.getErrorMsg(), LOG_CRITICAL);
		}
	}
	// search for best k
	else if (nlfSearch_g) {
		string scenario = dbCleanPrefix_g;

		fmNLFChoice(scenario);
	}
	// run test
	else if (runScenario_g){
		// initialize and name test scenarios
		// scenario -> results: 	TB    FM (Matrix)    FM (Tensor)    ...
		map<string, vector<double> > scenarioResults;
		for (unsigned int i = 1; i <= numOfScenarios_g; i++) {
			string strId = to_string(i);
			scenarioResults.insert(pair<string, vector<double> >(dbCleanPrefix_g+"_" + strId, vector<double>()));
			scenarioResults.insert(pair<string, vector<double> >(dbPrefix_g+"_" + strId, vector<double>()));
		}



		{
			ifstream inFile(dbCleanPrefix_g + DB_FILE_EXTENSION);
			if (!inFile.is_open()) {
				Logger::getInstance()->log("file '" + dbCleanPrefix_g + DB_FILE_EXTENSION + "' does not exist, cleaning dataset ...", LOG_DEBUG);
				try {
					cleanDataset(dbPrefix_g, 20);
				}
				catch (MyException e) {
					cout << e.getErrorMsg() << endl;
					cout << "make sure that there is a file '" + dbPrefix_g + DB_FILE_EXTENSION + "' present in yourr working directory!!" << endl;
					return 1;
				}
				Logger::getInstance()->log("clean dataset created!", LOG_DEBUG);

				
				Logger::getInstance()->log("creating train and test files ...", LOG_DEBUG);
				divideDataIntoTrainAndTestData(dbPrefix_g, numOfScenarios_g, 80);
				divideDataIntoTrainAndTestData(dbCleanPrefix_g, numOfScenarios_g, 80);
				Logger::getInstance()->log("train and test files created!", LOG_DEBUG);
			}
		}

		
		// initialize vector with column headings for result representation
		vector<string> methods;
		methods.push_back("TB");
		methods.push_back("FM_mat");
		methods.push_back("FM_ten");
		methods.push_back("FM_ten_attr");

		// initialize FM tuning parameters to run
		// you should initialize the fm predictor via the appropriate file for that data representation
		/*
		TatortFMPredictor fmPredictor;
		fmPredictor.setAlgorithm(initAlgo_g);
		fmPredictor.setIterations(initIter_g);
		fmPredictor.setStdev(initStdev_g);
		fmPredictor.setRegulation(initReg_g);
		fmPredictor.setLearningRate(initLr_g);
		*/


		readPredictorsFromFile();

		map<string, vector<double> >::iterator scenarioIter;
		vector<thread> threads;
		for (scenarioIter = scenarioResults.begin(); scenarioIter != scenarioResults.end(); scenarioIter++) {
			//testScenario(&scenarioIter->first, &scenarioIter->second);
			threads.push_back(thread(testScenario, &scenarioIter->first, &scenarioIter->second));
		}

		for (int i = 0; i < threads.size(); i++) {
			threads[i].join();
		}


		vector<double> means = computeMeanForResults(&scenarioResults);
		


		string strResults = resultsToStringHuman(&scenarioResults, &methods, &means);
		writeToFile("scenario_results.dat", strResults);
		Logger::getInstance()->log(strResults, LOG_INFO);
		strResults = resultsToString(&scenarioResults, &methods, &means);
		writeToFile("scenario_results.csv", strResults);
		
		//cout << strResults << endl;
	
	}
	else {
		
		// these methods should be called on a 'fresh' dataset (without user and detective ids)
		//unsigned int userIndex = 0;
		//addIdColumn(dbPrefix_g, userIndex, "UserID");
		//unsigned int detectiveIndex = 4;
		//addIdColumn(dbPrefix_g, detectiveIndex, "ErmittlerID");
		//completeViewersAndQuotes(dbPrefix_g, 8, 9);
		
	}

	clock_t end = clock();
	double elapsedSeconds = double(end - begin) / CLOCKS_PER_SEC;
	

	Logger::getInstance()->log("computation took '"+ to_string(elapsedSeconds) +"' seconds!", LOG_INFO);



	return 0;
}





void readPredictorsFromFile() {
	ifstream infile(matPredFile_g);
	if (!infile.is_open()) {
		Logger::getInstance()->log("file '" + matPredFile_g + "' does not exist!", LOG_CRITICAL);
		exit(1);
	}
	string algo, reg;
	int iter;
	double stdev, lr;
	int w0, w, k;
	infile >> algo;
	infile >> iter;
	infile >> stdev;
	infile >> reg;
	infile >> lr;
	infile >> w0;
	infile >> w;
	infile >> k;
	matrixPredictor_g.setAlgorithm(algo);
	matrixPredictor_g.setIterations(iter);
	matrixPredictor_g.setStdev(stdev);
	matrixPredictor_g.setRegulation(reg);
	matrixPredictor_g.setLearningRate(lr);
	matrixPredictor_g.parametersToUse(w0, w, k);
	infile.close();
	infile.clear();

	infile.open(tenPredFile_g);
	if (!infile.is_open()) {
		Logger::getInstance()->log("file '" + tenPredFile_g + "' does not exist!", LOG_CRITICAL);
		exit(1);
	}
	infile >> algo;
	infile >> iter;
	infile >> stdev;
	infile >> reg;
	infile >> lr;
	infile >> w0;
	infile >> w;
	infile >> k;
	tensorPredictor_g.setAlgorithm(algo);
	tensorPredictor_g.setIterations(iter);
	tensorPredictor_g.setStdev(stdev);
	tensorPredictor_g.setRegulation(reg);
	tensorPredictor_g.setLearningRate(lr);
	tensorPredictor_g.parametersToUse(w0, w, k);
	infile.close();
	infile.clear();

	infile.open(tenAttPredFile_g);
	if (!infile.is_open()) {
		Logger::getInstance()->log("file '" + tenAttPredFile_g + "' does not exist!", LOG_CRITICAL);
		exit(1);
	}
	infile >> algo;
	infile >> iter;
	infile >> stdev;
	infile >> reg;
	infile >> lr;
	infile >> w0;
	infile >> w;
	infile >> k;
	tensorAttributesPredictor_g.setAlgorithm(algo);
	tensorAttributesPredictor_g.setIterations(iter);
	tensorAttributesPredictor_g.setStdev(stdev);
	tensorAttributesPredictor_g.setRegulation(reg);
	tensorAttributesPredictor_g.setLearningRate(lr);
	tensorAttributesPredictor_g.parametersToUse(w0, w, k);
	infile.close();
	infile.clear();

}



TatortFMPredictor readPredictorFromFile(string filename) {
	ifstream infile(filename);
	if (!infile.is_open()) {
		Logger::getInstance()->log("file '" + filename+ "' does not exist!", LOG_CRITICAL);
		exit(1);
	}

	TatortFMPredictor pred;
	string algo, reg;
	int iter;
	double stdev, lr;
	int w0, w, k;
	infile >> algo;
	infile >> iter;
	infile >> stdev;
	infile >> reg;
	infile >> lr;
	infile >> w0;
	infile >> w;
	infile >> k;
	pred.setAlgorithm(algo);
	pred.setIterations(iter);
	pred.setStdev(stdev);
	pred.setRegulation(reg);
	pred.setLearningRate(lr);
	pred.parametersToUse(w0, w, k);

	return pred;
}



// runs a bunch of parameter combination, not per algorithm but within in a range of iterations to take. this thread function is a
// much better parallelization of parameter testing, becausle all threads have approximately the same number of iterations to compute
void fmParallelParameterChoicePerNLF(string trainFilename, string testFilename, ThreadData_t *td) {

	// run SGD with different tuning params \Theta = {standarddeviation, iterations, regulation, learnrate}
	// run ALS with different tuning params \Theta = {standarddeviation, iterations, regulation}
	// run MCMC with different tuning params \Theta = {standarddeviation, iterations}

	Logger::getInstance()->log("experimentally searching for the best model parameter choices ...", LOG_DEBUG);

	int errorCount = 0;

	string paramFile_sgd = "param_search_sgd.dat";
	string paramFile_als = "param_search_als.dat";
	string paramFile_mcmc = "param_search_mcmc.dat";

	string bestParameterFile = "best_param.dat";
	string bestParameterFile_mcmc = "best_param_mcmc.dat";
	string bestParameterFile_als = "best_param_als.dat";
	string bestParameterFile_sgd = "best_param_sgd.dat";

	string predFilename = "pred_result_"+ to_string(td->threadID);


	if(mcmc_g) {
		mcmcFileMutex_g.lock();
		writeToFile(paramFile_mcmc, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		writeToFile(bestParameterFile_mcmc, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		mcmcFileMutex_g.unlock();
	}


	if(als_g) {
		alsFileMutex_g.lock();
		writeToFile(paramFile_als, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		writeToFile(bestParameterFile_als, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		alsFileMutex_g.unlock();
	}

	if(sgd_g) {
		sgdFileMutex_g.lock();
		writeToFile(paramFile_sgd, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		writeToFile(bestParameterFile_sgd, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		sgdFileMutex_g.unlock();
	}


	//runFMParser(trainFilename, testFilename, DATA_UED_TENSOR);


	if (mcmc_g || als_g || sgd_g) {
		for (int k = td->start; k <= td->stop; k += td->step) {
			TatortFMPredictor currFmPredictor;
			TatortFMPredictor bestAlsPredictor;
			TatortFMPredictor bestSgdPredictor;
			TatortFMPredictor bestMcmcPredictor;

			double bestAlsResult = numeric_limits<double>::max();
			double bestSgdResult = numeric_limits<double>::max();
			double bestMcmcResult = numeric_limits<double>::max();
			currFmPredictor.setNumOfLatentFactors(k);


			for (int iters = iterStart_g; iters <= iterStop_g; iters += iterStep_g) {
				
				currFmPredictor.setIterations(iters);

				for (double stdev = stdevStart_g; stdev <= stdevStop_g; stdev += stdevStep_g) {
					currFmPredictor.setStdev(stdev);
					currFmPredictor.setAlgorithm("mcmc");

					if (mcmc_g)
						checkParams(trainFilename, testFilename, predFilename, &currFmPredictor, &bestMcmcPredictor, &bestMcmcResult, paramFile_mcmc, &mcmcFileMutex_g);

					if (als_g || sgd_g) {
						for (double reg = regStart_g; reg <= regStop_g; reg += regStep_g) {
							currFmPredictor.setRegulation(to_string(reg));
							currFmPredictor.setAlgorithm("als");

							if (als_g)
								checkParams(trainFilename, testFilename, predFilename, &currFmPredictor, &bestAlsPredictor, &bestAlsResult, paramFile_als, &alsFileMutex_g);

							if (sgd_g) {
								for (double lr = lrStart_g; lr <= lrStop_g; lr += lrStep_g) {
									currFmPredictor.setLearningRate(lr);
									currFmPredictor.setAlgorithm("sgd");

									checkParams(trainFilename, testFilename, predFilename, &currFmPredictor, &bestSgdPredictor, &bestSgdResult, paramFile_sgd, &sgdFileMutex_g);
								}
							}
						}
					}
				}
			}

			string line;

			if (mcmc_g) {
				line = predictionToString(&bestMcmcPredictor, bestMcmcResult);
				mcmcFileMutex_g.lock();
				appendLineToFile(bestParameterFile_mcmc, line);
				mcmcFileMutex_g.unlock();
			}

			if (als_g) {
				line = predictionToString(&bestAlsPredictor, bestAlsResult);
				alsFileMutex_g.lock();
				appendLineToFile(bestParameterFile_als, line);
				alsFileMutex_g.unlock();
			}

			if (sgd_g) {
				line = predictionToString(&bestSgdPredictor, bestSgdResult);
				sgdFileMutex_g.lock();
				appendLineToFile(bestParameterFile_sgd, line);
				sgdFileMutex_g.unlock();
			}
		}
	}



	Logger::getInstance()->log("choosing parameters done! got '" + to_string(errorCount) + "' errors!", LOG_DEBUG);


}



// runs a bunch of parameter combination, not per algorithm but within in a range of iterations to take. this thread function is a
// much better parallelization of parameter testing, becausle all threads have approximately the same number of iterations to compute
void fmNLFChoice(string scenario) {
	Logger::getInstance()->log("experimentally searching for the best choice of nlf ...", LOG_DEBUG);

	scenario += "_1";
	string trainFilename = scenario + trainSuffix_g;
	string testFilename = scenario + testSuffix_g;
	string predFilename = "pred_result";

	int errorCount = 0;

	string resFile = "serach_k.dat";
	


	runFMParser(scenario+trainSuffix_g, scenario+testSuffix_g, dataRepresentation_g);
	

	writeToFile(resFile, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");


	for(int k = nlfStart_g; k <= nlfStop_g; k += nlfStep_g) {
		nlfPredictor_g.setNumOfLatentFactors(k);
		double res = numeric_limits<double>::max();
		try {
			res = runFMPredictor(trainFilename, testFilename, predFilename, nlfPredictor_g);
			
			
			ostringstream os;
			os << res << delimiter_g;
			os << predictorToString(nlfPredictor_g);


			appendLineToFile(resFile, os.str());
		}
		catch (MyException e) {
			Logger::getInstance()->log(e.getErrorMsg(), LOG_ERROR);
		}
	}


	Logger::getInstance()->log("choosing parameters done! got '" + to_string(errorCount) + "' errors!", LOG_DEBUG);

}



string predictorToString(TatortFMPredictor fmPred) {
	ostringstream os;
	os << fmPred.getAlgorithm() << delimiter_g;
	os << fmPred.getNumOfLatentFactors() << delimiter_g;
	os << fmPred.getIterations() << delimiter_g;
	os << fmPred.getStdev() << delimiter_g;
	os << fmPred.getRegulation() << delimiter_g;
	os << fmPred.getLearningRate();
	return os.str();
}



vector<double> computeMeanForResults(map<string, vector<double> >* predictionResults) {

	vector<double> means;
	map<string, vector<double> >::iterator scenarioIter = predictionResults->begin();
	if (scenarioIter == predictionResults->end())
		return means;

	int numberOfScenarios = 0;
	for (unsigned int i = 0; i < scenarioIter->second.size(); i++) {
		means.push_back(scenarioIter->second[i]);
	}
	numberOfScenarios++;

	for (++scenarioIter; scenarioIter != predictionResults->end(); scenarioIter++) {
		for (unsigned int i = 0; i < scenarioIter->second.size(); i++) {
			means[i] += scenarioIter->second[i];
		}
		numberOfScenarios++;
	}

	for (unsigned i = 0; i < means.size(); i++) {
		means[i] /= numberOfScenarios;
	}

	return means;

}


// initiates parameter testing for the given scenario. the number of threads given determines if the serial (1 or lower) or parallel (> 1) variant is used
void searchingOptimalParams(string scenario, unsigned int numOfThreads) {
	if (numOfThreads >= 1) {

		//divideDataIntoTrainAndTestData(scenario, 1, 80);
		scenario += "_1";
		runFMParser(scenario+trainSuffix_g, scenario+testSuffix_g, dataRepresentation_g);

		if(!fileExist(scenario+trainSuffix_g+LIBFM_FILE_EXTENSION))
			throw MyException("file '"+scenario+trainSuffix_g+LIBFM_FILE_EXTENSION+"' does not exist!");
		if(!fileExist(scenario+testSuffix_g+LIBFM_FILE_EXTENSION))
			throw MyException("file '"+scenario+testSuffix_g+LIBFM_FILE_EXTENSION+"' does not exist!");
		if(!fileExist(scenario+targetSuffix_g+DB_FILE_EXTENSION))
			throw MyException("file '"+scenario+targetSuffix_g+LIBFM_FILE_EXTENSION+"' does not exist!");

		/*
		method 1:
		// working but inefficient way of parallelization, because sgd thread has a lot more to do as als and mcmc thread
		{
			TatortFMPredictor fmPredictor_mcmc;
			TatortFMPredictor fmPredictor_als;
			TatortFMPredictor fmPredictor_sgd;

			thread sgd(fmParallelParameterChoicePerAlgorithm, scenario + "_train", scenario + "_test", "sgd", &fmPredictor_sgd);
			thread als(fmParallelParameterChoicePerAlgorithm, scenario + "_train", scenario + "_test", "als", &fmPredictor_als);
			thread mcmc(fmParallelParameterChoicePerAlgorithm, scenario + "_train", scenario + "_test", "mcmc", &fmPredictor_mcmc);

			mcmc.join();
			als.join();
			sgd.join();
		}
		*/

		// better way of parallelization: computing number of iterations and distribute over all threads. so every thread
		// computes sgd, als, and mcmc of its range of iterations
		vector<thread> threads;
		vector<ThreadData_t *> threadData;


		// choose range of outer loop
		int workStart = iterStart_g;
		int workStop = iterStop_g;
		int workStep = iterStep_g;

		if(includeNLF_g) {
			workStart = nlfStart_g;
			workStop = nlfStop_g;
			workStep = nlfStep_g;
		}

		// choose iteration range
		/*
		int iterStart = iterStart_g;
		int iterStop = iterStop_g;
		int iterStep = iterStep_g;
		*/

		// compute work to do)
		unsigned int workToDo = 0;
		for (int w = workStart; w <= workStop; w += workStep) {
			workToDo += w;
		}


		// compute fraction, that every thread has to do
		int workPerThread = workToDo / numOfThreads;

		for (unsigned int i = 0; i < numOfThreads; ++i) {
			
			// create data for current thread
			ThreadData_t *td = new ThreadData_t;
			td->threadID = i;
			// init stepsize
			td->step = workStep;

			// init stop for current thread
			if (i == 0) {
				td->stop = workStop;
			}
			else {
				td->stop = threadData[i - 1]->start - workStep;
			}

			// the last thread should do the rest
			if (i == (numOfThreads - 1)) {
				td->start = workStart;
			}
			else {
				// compute end of current thread
				td->start = td->stop;
				int workForCurrentThread = td->stop;
				do {
					td->start -= workStep;
					workForCurrentThread = workForCurrentThread + td->start;
				} while(workForCurrentThread <= workPerThread);
				// for better distribution
				td->start += workStep;
			}

			threadData.push_back(td);
			if(includeNLF_g)
				threads.push_back(thread(fmParallelParameterChoicePerNLF, scenario + trainSuffix_g, scenario + testSuffix_g, threadData[i]));
			else
				threads.push_back(thread(fmParallelParameterChoicePerIteration, scenario + trainSuffix_g, scenario + testSuffix_g, threadData[i]));
		}


		for (unsigned int i = 0; i < threads.size(); i++) {
			threads[i].join();
			delete threadData[i];
		}

	}
	else {
		Logger::getInstance()->log("invalid number of threads!", LOG_ERROR);
	}
}






// runs a bunch of parameter combination, not per algorithm but within in a range of iterations to take. this thread function is a
// much better parallelization of parameter testing, becausle all threads have approximately the same number of iterations to compute
void fmParallelParameterChoicePerIteration(string trainFilename, string testFilename, ThreadData_t *td) {

	// run SGD with different tuning params \Theta = {standarddeviation, iterations, regulation, learnrate}
	// run ALS with different tuning params \Theta = {standarddeviation, iterations, regulation}
	// run MCMC with different tuning params \Theta = {standarddeviation, iterations}

	Logger::getInstance()->log("experimentally searching for the best model parameter choices ...", LOG_DEBUG);

	int errorCount = 0;

	string paramFile_sgd = "param_search_sgd.dat";
	string paramFile_als = "param_search_als.dat";
	string paramFile_mcmc = "param_search_mcmc.dat";

	string bestParameterFile = "best_param.dat";
	string bestParameterFile_mcmc = "best_param_mcmc.dat";
	string bestParameterFile_als = "best_param_als.dat";
	string bestParameterFile_sgd = "best_param_sgd.dat";

	string predFilename = "pred_result_"+ to_string(td->threadID);


	if(mcmc_g) {
		mcmcFileMutex_g.lock();
		writeToFile(paramFile_mcmc, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		writeToFile(bestParameterFile_mcmc, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		mcmcFileMutex_g.unlock();
	}


	if(als_g) {
		alsFileMutex_g.lock();
		writeToFile(paramFile_als, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		writeToFile(bestParameterFile_als, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		alsFileMutex_g.unlock();
	}

	if(sgd_g) {
		sgdFileMutex_g.lock();
		writeToFile(paramFile_sgd, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		writeToFile(bestParameterFile_sgd, "result" + delimiter_g + "algo" + delimiter_g + "k" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		sgdFileMutex_g.unlock();
	}


	//runFMParser(trainFilename, testFilename, DATA_UED_TENSOR);

	// 10 * 10 * 10 * 20
	// 5 * 10 * 5 * 5
	if (mcmc_g || als_g || sgd_g) {
		for (int iters = td->start; iters <= td->stop; iters += td->step) {
			TatortFMPredictor currFmPredictor;
			TatortFMPredictor bestAlsPredictor;
			TatortFMPredictor bestSgdPredictor;
			TatortFMPredictor bestMcmcPredictor;

			double bestAlsResult = numeric_limits<double>::max();
			double bestSgdResult = numeric_limits<double>::max();
			double bestMcmcResult = numeric_limits<double>::max();
			currFmPredictor.setNumOfLatentFactors(nlf_g);
			currFmPredictor.setIterations(iters);

			for (double stdev = stdevStart_g; stdev <= stdevStop_g; stdev += stdevStep_g) {
				currFmPredictor.setStdev(stdev);
				currFmPredictor.setAlgorithm("mcmc");

				if (mcmc_g)
					checkParams(trainFilename, testFilename, predFilename, &currFmPredictor, &bestMcmcPredictor, &bestMcmcResult, paramFile_mcmc, &mcmcFileMutex_g);

				if (als_g || sgd_g) {
					for (double reg = regStart_g; reg <= regStop_g; reg += regStep_g) {
						currFmPredictor.setRegulation(to_string(reg));
						currFmPredictor.setAlgorithm("als");

						if (als_g)
							checkParams(trainFilename, testFilename, predFilename, &currFmPredictor, &bestAlsPredictor, &bestAlsResult, paramFile_als, &alsFileMutex_g);

						if (sgd_g) {
							for (double lr = lrStart_g; lr <= lrStop_g; lr += lrStep_g) {
								currFmPredictor.setLearningRate(lr);
								currFmPredictor.setAlgorithm("sgd");

								checkParams(trainFilename, testFilename, predFilename, &currFmPredictor, &bestSgdPredictor, &bestSgdResult, paramFile_sgd, &sgdFileMutex_g);
							}
						}
					}
				}
			}

			string line;

			if (mcmc_g) {
				line = predictionToString(&bestMcmcPredictor, bestMcmcResult);
				mcmcFileMutex_g.lock();
				appendLineToFile(bestParameterFile_mcmc, line);
				mcmcFileMutex_g.unlock();
			}

			if (als_g) {
				line = predictionToString(&bestAlsPredictor, bestAlsResult);
				alsFileMutex_g.lock();
				appendLineToFile(bestParameterFile_als, line);
				alsFileMutex_g.unlock();
			}

			if (sgd_g) {
				line = predictionToString(&bestSgdPredictor, bestSgdResult);
				sgdFileMutex_g.lock();
				appendLineToFile(bestParameterFile_sgd, line);
				sgdFileMutex_g.unlock();
			}


		}
	}



	Logger::getInstance()->log("choosing parameters done! got '" + to_string(errorCount) + "' errors!", LOG_DEBUG);
}




















// creates a string from the given FM Predictor, the best result and the number of iterations, this result belongs to
string predictionToString(TatortFMPredictor *bestFmPredictor, double bestResult) {
	
	string algo = bestFmPredictor->getAlgorithm();
	ostringstream os;
	os << bestResult << delimiter_g;
	os << bestFmPredictor->getAlgorithm() << delimiter_g;
	os << bestFmPredictor->getNumOfLatentFactors() << delimiter_g;
	os << bestFmPredictor->getIterations() << delimiter_g;
	os << bestFmPredictor->getStdev();
	os << delimiter_g << bestFmPredictor->getRegulation();
	os << delimiter_g << bestFmPredictor->getLearningRate();
	return os.str();
}














void checkParams(string trainFilename, string testFilename, string predFilename, TatortFMPredictor *currFmPredictor, TatortFMPredictor *bestFmPredictor, double *bestResult, string outFilename, mutex *fileMutex) {

	double res = numeric_limits<double>::max();
	try {
		//res = fmTrainAndTest(trainFilename, testFilename, delimiter, "param_pred", *fmPredictor, DATA_UED_TENSOR);
		res = runFMPredictor(trainFilename, testFilename, predFilename, *currFmPredictor);
				
		string line = predictionToString(currFmPredictor, res);
		
		// check if file is free for writing write
		if (fileMutex != NULL)
			fileMutex->lock();
		
		appendLineToFile(outFilename, line);
		
		if (fileMutex != NULL)
			fileMutex->unlock();
	}
	catch (MyException e) {
		Logger::getInstance()->log(e.getErrorMsg(), LOG_ERROR);
	}
	if (res < *bestResult) {
		*bestResult = res;
		bestFmPredictor->copyFrom(currFmPredictor);
	}
}


void appendLineToFile(string filename, string text) {
	fstream file;
	file.open(filename, fstream::app);
	if (!file.is_open())
		throw MyException("file '"+filename+"' could not be opened!");

	file << text << endl;
}


// writes text to file (filename)
void writeToFile(string filename, string text) {
	ofstream of(filename.c_str());

	if(!of.is_open())
		throw MyException("cannot open file '"+ filename +"'!");

	of << text;
}



// creates a string with results of all test scenarios (human readable)
string resultsToStringHuman(map<string, vector<double> >* predictionResults, vector<string>* methods, vector<double>* means) {
	map<string, vector<double> >::iterator pqIter;

	int scenarioColumnWidth = 20;
	int columnWidth = 15;

	ostringstream os;

	os << "====================================================================" << endl;
	os << "                    Prediction Results (MAE)                        " << endl;
	os << "====================================================================" << endl;
	os << setw(scenarioColumnWidth);
	os << "Scenario";
	
	string hline = "";
	for(int i = 0; i < scenarioColumnWidth; i++)
		hline += "-";

	for (unsigned int i = 0; i < methods->size(); i++)
	{
		os << setw(columnWidth) << (*methods)[i];
		for(int i = 0; i < columnWidth; i++)
			hline += "-";
	}
	os << endl << hline << endl;

	for(pqIter = predictionResults->begin(); pqIter != predictionResults->end(); pqIter++) {
		os << setw(scenarioColumnWidth) << pqIter->first;
		for (unsigned int i = 0; i < pqIter->second.size(); i++)
		{
			os << setw(columnWidth) << pqIter->second[i];
		}
		os << endl;
	}

	os << setw(scenarioColumnWidth) << "means";
	for (unsigned int i = 0; i < means->size(); i++)
	{
		os << setw(columnWidth) << (*means)[i];
	}

	os << endl;


	return os.str();
}


// creates a string with results of all test scenarios
string resultsToString(map<string, vector<double> >* predictionResults, vector<string>* methods, vector<double>* means) {
	map<string, vector<double> >::iterator pqIter;

	//int scenarioColumnWidth = 20;
	//int columnWidth = 15;

	ostringstream os;

	os << "Scenario";

	for (unsigned int i = 0; i < methods->size(); i++)
	{
		os << (*methods)[i];
		if (i != methods->size() - 1)
			os << delimiter_g;
	}

	for (pqIter = predictionResults->begin(); pqIter != predictionResults->end(); pqIter++) {
		os << pqIter->first << delimiter_g;
		for (unsigned int i = 0; i < pqIter->second.size(); i++)
		{
			os << pqIter->second[i];
			if (i != pqIter->second.size() - 1)
				os << delimiter_g;
		}
		os << endl;
	}

	os << "means" << delimiter_g;
	for (unsigned int i = 0; i < means->size(); i++)
	{
		os << (*means)[i];
		if (i != means->size() - 1)
			os << delimiter_g;
	}

	os << endl;

	return os.str();
}


// predict the given scenario with all implemented algorithms
void testScenario(const string *scenarioName, vector<double>* prediction) {
	
	// run predictions and remember results in map
	vector<double> results;
	Logger::getInstance()->log("train and test '" + *scenarioName + "' ...", LOG_DEBUG);
	prediction->push_back(tendencyTrainAndTest(*scenarioName + trainSuffix_g, *scenarioName + testSuffix_g, *scenarioName + predSuffix_g));

	prediction->push_back(fmTrainAndTest(*scenarioName + trainSuffix_g, *scenarioName + testSuffix_g, *scenarioName + predSuffix_g, matrixPredictor_g, DATA_UE_MATRIX));
	prediction->push_back(fmTrainAndTest(*scenarioName + trainSuffix_g, *scenarioName + testSuffix_g, *scenarioName + predSuffix_g, tensorPredictor_g, DATA_UED_TENSOR));
	prediction->push_back(fmTrainAndTest(*scenarioName + trainSuffix_g, *scenarioName + testSuffix_g, *scenarioName + predSuffix_g, tensorAttributesPredictor_g, DATA_UED_TENSOR_PLUS_ATTRIBUTES));

	Logger::getInstance()->log("done with scenario '"+ *scenarioName +"' ...", LOG_DEBUG);
}


// train and test data with FM and returns MAE
double fmTrainAndTest(string trainFilename, string testFilename, string predFilename, TatortFMPredictor fmPredictor, int dataRepresentation) {

	double res = -1;

	//if(!fileExist(trainFilename+LIBFM_FILE_EXTENSION) || !fileExist(testFilename+LIBFM_FILE_EXTENSION))
		runFMParser(trainFilename, testFilename, dataRepresentation);
	//else
	//	Logger::getInstance()->log("files "+trainFilename+LIBFM_FILE_EXTENSION+" and "+testFilename+LIBFM_FILE_EXTENSION+" already exist. skipping creating and parsing them!", LOG_INFO);

	res = runFMPredictor(trainFilename, testFilename, predFilename, fmPredictor);
	
	return res;
}


// calls overloaded function 
void runFMParser(string trainFilename, string testFilename, int dataRepresentation) {
	
	runFMParser(trainFilename, testFilename, dataRepresentation, trainFilename, testFilename);
}


// parses the csv inputfiles in the given (dataRepresentration) FM file format
void runFMParser(string inTrainFilename, string inTestFilename, int dataRepresentation, string outTrainFilename, string outTestFilename) {
	TatortFMParser fmParser;

	switch (dataRepresentation)
	{
	case DATA_UE_MATRIX:
		fmParser.convertDataToMatrix(inTrainFilename + DB_FILE_EXTENSION, inTestFilename + DB_FILE_EXTENSION, delimiter_g, outTrainFilename + LIBFM_FILE_EXTENSION, outTestFilename + LIBFM_FILE_EXTENSION, true);
		break;

	case DATA_UED_TENSOR:
		fmParser.convertDataToTensor(inTrainFilename + DB_FILE_EXTENSION, inTestFilename + DB_FILE_EXTENSION, delimiter_g, outTrainFilename + LIBFM_FILE_EXTENSION, outTestFilename + LIBFM_FILE_EXTENSION, true);
		break;
		
	case DATA_UED_TENSOR_PLUS_ATTRIBUTES:
		fmParser.convertDataToTensorPlusAttributes(inTrainFilename + DB_FILE_EXTENSION, inTestFilename + DB_FILE_EXTENSION, delimiter_g, attributeIndicesToUse_g, outTrainFilename + LIBFM_FILE_EXTENSION, outTestFilename + LIBFM_FILE_EXTENSION, true);
		break;

	default:
		Logger::getInstance()->log("unknown parsing option for FMParser", LOG_ERROR);
		break;
	}
}



double runFMPredictor(string trainFilename, string testFilename, string predFilename, TatortFMPredictor fmPredictor) {
	// run FM predictor and compute MAE
	return fmPredictor.trainAndTest(trainFilename + LIBFM_FILE_EXTENSION, testFilename + LIBFM_FILE_EXTENSION, predFilename + LIBFM_FILE_EXTENSION);
}




// constructs a Tendency predictor, trains it and computes MAE for predicted testdata
double tendencyTrainAndTest(string trainFilename, string testFilename, string predFilename) {
	TatortTendencyPredictor tp;

	// train tendency predictor (compute means, etc.)
	tp.train(trainFilename + DB_FILE_EXTENSION, delimiter_g, true);
	double result = -1;

	try {
		// predict and compute MAE
		result = tp.test(testFilename + DB_FILE_EXTENSION, delimiter_g, true, predFilename + DB_FILE_EXTENSION);
	}
	catch (MyException e) {
		cout << e.getErrorMsg() << endl;
	}

	return result;
}




// removes all users with less than the given number of ratings
void cleanDataset(string filename, int userRatingThreshold) {
	TatortTendencyParser tp;

	tp.cleanData(filename + DB_FILE_EXTENSION, delimiter_g, true, filename + "_clean"+ DB_FILE_EXTENSION, userRatingThreshold);
}




void addIdColumn(string filename, unsigned int columnId, string newColumnTitel) {
	Parser p;
	try {
		p.parseFile(filename + DB_FILE_EXTENSION, delimiter_g, true);
		p.addIdColumnToFile(filename + DB_FILE_EXTENSION, columnId, newColumnTitel, delimiter_g);
	}
	catch (MyException e) {
		cout << e.getErrorMsg() << endl;
	}
}

// adds user id to tatort db (csv file)
// deprecated: should NOT be used any more
void addUserIdAndDetectiveId(string filename) {
	Parser p;
	try {
		p.parseFile(filename + DB_FILE_EXTENSION, delimiter_g, true);
		p.addIdColumnToFile(filename + DB_FILE_EXTENSION, 0, "UserID", delimiter_g);
	}
	catch (MyException e) {
		cout << e.getErrorMsg() << endl;
	}

	p.clear();
	
	try {
		p.parseFile(filename + DB_FILE_EXTENSION, delimiter_g, true);
		p.addIdColumnToFile(filename + DB_FILE_EXTENSION, 4, "ErmittlerID", delimiter_g);
	}
	catch (MyException e) {
		cout << e.getErrorMsg() << endl;
	}
}



void completeViewersAndQuotes(string filename, int viewerIndex, int quoteIndex) {
	Parser p;


	p.addMissingViewersAndQuotes(filename+DB_FILE_EXTENSION, delimiter_g, viewerIndex, quoteIndex);
	
}



// randomly selects trainpercentage entries from database and assumes that these are train data. the rest is assumed to be test data.
void divideDataIntoTrainAndTestData(string sourceFilename, int count, int trainPercentage) {
	Parser p;
	try {
		for (int i = 1; i <= count; i++) {
			string trainSuffix = "_" + to_string(i) + trainSuffix_g + DB_FILE_EXTENSION;
			string testSuffix = "_" + to_string(i) + testSuffix_g + DB_FILE_EXTENSION;
			string targetSuffix = "_" + to_string(i) + targetSuffix_g + DB_FILE_EXTENSION;
			p.divideLinesTrainAndTest(sourceFilename + DB_FILE_EXTENSION, true, trainPercentage, sourceFilename +trainSuffix, sourceFilename + testSuffix, sourceFilename + targetSuffix);
		}
	}
	catch (MyException e) {
		cout << e.getErrorMsg() << endl;
	}
}


bool fileExist(string filename) {
	ifstream file(filename);
	if(file.is_open())
		return true;
	else
		return false;
}



int parseCmdLineArguments(int argc, char **argv) {
	// parsing command line arguments
	for (int i = 0; i < argc; i++) {
		

		// parameter choice
		if(!string("-search").compare(argv[i])) {
			parameterSearch_g = true;
			if(argc > i+1) {
				includeNLF_g = stoi(argv[++i]);
			} else {
				Logger::getInstance()->log("missing param for option '" + string(argv[i]) + "'!", LOG_ERROR);
				showHelp();
				return 1;
			}
		}
		else if(!string("-threads").compare(argv[i])) {
			if(argc > i+1) {
				numOfThreads_g = stoi(argv[++i]);
			} else {
				Logger::getInstance()->log("missing param for option '" + string(argv[i]) + "'!", LOG_ERROR);
				showHelp();
				return 1;
			}
		}
		else if(!string("-algos").compare(argv[i])) {
			if(argc > i+1) {
				vector<string> params = StringTokenizer::justTokenize(string(argv[++i]), ",");
				if(params.size() == 3) {
					if(stoi(params[0]) == 1) mcmc_g = true;
					if(stoi(params[1]) == 1) als_g = true;
					if(stoi(params[2]) == 1) sgd_g = true;
				}
				else {
					Logger::getInstance()->log("wrong param format for option '" + string(argv[i-1]) + "'!", LOG_ERROR);
					showHelp();
					return 1;
				}
			} else {
				Logger::getInstance()->log("missing param for option '" + string(argv[i]) + "'!", LOG_ERROR);
				showHelp();
				return 1;
			}
		}
		else if(!string("-iter").compare(argv[i])) {
			if(argc > i+1) {
				vector<string> params = StringTokenizer::justTokenize(string(argv[++i]), ",");
				if(params.size() == 3) {
					iterStart_g = stoi(params[0]);
					iterStop_g = stoi(params[1]);
					iterStep_g = stoi(params[2]);
				}
				else {
					Logger::getInstance()->log("wrong param format for option '" + string(argv[i-1]) + "'!", LOG_ERROR);
					showHelp();
					return 1;
				}
			} else {
				Logger::getInstance()->log("missing param for option '" + string(argv[i]) + "'!", LOG_ERROR);
				showHelp();
				return 1;
			}
		}
		else if(!string("-stdev").compare(argv[i])) {
			if(argc > i+1) {
				vector<string> params = StringTokenizer::justTokenize(string(argv[++i]), ",");
				if(params.size() == 3) {
					stdevStart_g = stod(params[0]);
					stdevStop_g = stod(params[1]);
					stdevStep_g = stod(params[2]);
				}
				else {
					Logger::getInstance()->log("wrong param format for option '" + string(argv[i-1]) + "'!", LOG_ERROR);
					showHelp();
					return 1;
				}
			} else {
				Logger::getInstance()->log("missing param for option '" + string(argv[i]) + "'!", LOG_ERROR);
				showHelp();
				return 1;
			}
		}
		else if(!string("-reg").compare(argv[i])) {
			if(argc > i+1) {
				vector<string> params = StringTokenizer::justTokenize(string(argv[++i]), ",");
				if(params.size() == 3) {
					regStart_g = stod(params[0]);
					regStop_g = stod(params[1]);
					regStep_g = stod(params[2]);
				}
				else {
					Logger::getInstance()->log("wrong param format for option '" + string(argv[i-1]) + "'!", LOG_ERROR);
					showHelp();
					return 1;
				}
			} else {
				Logger::getInstance()->log("missing param for option '" + string(argv[i]) + "'!", LOG_ERROR);
				showHelp();
				return 1;
			}
		}
		else if(!string("-lr").compare(argv[i])) {
			if(argc > i+1) {
				vector<string> params = StringTokenizer::justTokenize(string(argv[++i]), ",");
				if(params.size() == 3) {
					lrStart_g = stod(params[0]);
					lrStop_g = stod(params[1]);
					lrStep_g = stod(params[2]);
				}
				else {
					Logger::getInstance()->log("wrong param format for option '" + string(argv[i-1]) + "'!", LOG_ERROR);
					showHelp();
					return 1;
				}
			} else {
				Logger::getInstance()->log("missing param for option '" + string(argv[i]) + "'!", LOG_ERROR);
				showHelp();
				return 1;
			}
		}
		else if(!string("-k").compare(argv[i])) {
			if(argc > i+1) {
				nlfSearch_g = true;
				vector<string> params = StringTokenizer::justTokenize(string(argv[++i]), ",");
				if(params.size() == 3) {
					nlfStart_g = stoi(params[0]);
					nlfStop_g = stoi(params[1]);
					nlfStep_g = stoi(params[2]);
				}
				else {
					Logger::getInstance()->log("wrong param format for option '" + string(argv[i-1]) + "'!", LOG_ERROR);
					showHelp();
					return 1;
				}
			} else {
				Logger::getInstance()->log("missing param for option '" + string(argv[i]) + "'!", LOG_ERROR);
				showHelp();
				return 1;
			}
		}
		else if( !string("-dr").compare(argv[i]) ) {
			if(argc > i+1) {
				dataRepresentation_g = stoi(argv[++i]);
			} else {
				Logger::getInstance()->log("missing param for option '" + string(argv[i]) + "'!", LOG_ERROR);
				showHelp();
				return 1;
			}
		}










		// scenario testing
		else if(!string("-test").compare(argv[i])) {
			if(argc > i+1) {
				runScenario_g = true;
				numOfScenarios_g = stoi(argv[++i]);
			} else {
				Logger::getInstance()->log("missing param for option '" + string(argv[i]) + "'!", LOG_ERROR);
				showHelp();
				return 1;
			}
		}
		else if (!string("-attr").compare(argv[i])) {
			if (argc > i + 1) {
				attributeIndicesToUse_g.clear();
				vector<string> params = StringTokenizer::justTokenize(string(argv[++i]), ",");
				for (unsigned int i = 0; i < params.size(); i++)
					attributeIndicesToUse_g.push_back(stoi(params[i]));
			}
			else {
				Logger::getInstance()->log("missing param for option '" + string(argv[i]) + "'!", LOG_ERROR);
				showHelp();
				return 1;
			}
		}





		// general options
		else if( (!string("-help").compare(argv[i])) || (!string("-h").compare(argv[i])) ) {
			//cout << argv[i] << " is present!" << endl;
			help_g = true;
		}
	}

	return 0;
}







// OLD VERSION --- DONT USE THIS FUNCTION
// threadfunction: runs a bunch of parameter combinations with only ONE algorithm (per thread)
/*
void fmParallelParameterChoicePerAlgorithm(string trainFilename, string testFilename, string algorithm, TatortFMPredictor *bestFmPredictor) {
	Logger::getInstance()->log("experimentally searching for the best model parameter choices ...", LOG_DEBUG);

	int errorCount = 0;

	TatortFMPredictor currFmPredictor;


	string paramFile_sgd = "param_search_sgd.dat";
	string paramFile_als = "param_search_als.dat";
	string paramFile_mcmc = "param_search_mcmc.dat";

	string bestParameterFile = "best_param.dat";
	string bestParameterFile_mcmc = "best_param_mcmc.dat";
	string bestParameterFile_als = "best_param_als.dat";
	string bestParameterFile_sgd = "best_param_sgd.dat";

	string predFile_sgd = "pred_sgd";
	string predFile_als = "pred_als";
	string predFile_mcmc = "pred_mcmc";
	
	

	double bestResult = numeric_limits<double>::max();

	//runFMParser(trainFilename, testFilename, delimiter, DATA_UED_TENSOR);

	if (algorithm == "mcmc") {
		writeToFile(paramFile_mcmc, "result" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		writeToFile(bestParameterFile_mcmc, "result" + delimiter_g + "iterations" + delimiter_g + "bset_iterations" + delimiter_g + "stdev");
		
		// 20 * 6 = 120
		for (int iters = 20; iters <= 400; iters += 20) {
			currFmPredictor.setIterations(iters);

			for (double stdev = 0.0; stdev <= 1.0; stdev += 0.2) {
				currFmPredictor.setStdev(stdev);
				currFmPredictor.setAlgorithm("mcmc");

				checkParams(trainFilename, testFilename, predFile_mcmc, &currFmPredictor, bestFmPredictor, &bestResult, paramFile_mcmc);
			}

			string line = predictionToString(bestFmPredictor, bestResult);
			
			appendLineToFile(bestParameterFile_mcmc, line);
			Logger::getInstance()->log(line, LOG_DEBUG);
		}
	}
	else if (algorithm == "als") {
		writeToFile(paramFile_als, "result" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		writeToFile(bestParameterFile_als, "result" + delimiter_g + "iterations" + delimiter_g + "bset_iterations" + delimiter_g + "stdev" + delimiter_g + "regulation");
		
		// 20 * 6 * 6 = 720
		for (int iters = 20; iters <= 400; iters += 20) {
			currFmPredictor.setIterations(iters);

			for (double stdev = 0.0; stdev <= 1.0; stdev += 0.2) {
				currFmPredictor.setStdev(stdev);

				for (double reg = 0.0; reg <= 1.0; reg += 0.2) {
					currFmPredictor.setRegulation(to_string(reg));
					currFmPredictor.setAlgorithm("als");

					checkParams(trainFilename, testFilename, predFile_als, &currFmPredictor, bestFmPredictor, &bestResult, paramFile_als);
				}
			}

			string line = predictionToString(bestFmPredictor, bestResult, iters);

			appendLineToFile(bestParameterFile_als, line);
			Logger::getInstance()->log(line, LOG_DEBUG);
		}
	}
	else if (algorithm == "sgd") {
		writeToFile(paramFile_sgd, "result" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		writeToFile(bestParameterFile_sgd, "result" + delimiter_g + "iterations" + delimiter_g + "bset_iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
		
		// 20 * 6 * 6 * 21 = 15120
		for (int iters = 20; iters <= 400; iters += 20) {
			currFmPredictor.setIterations(iters);

			for (double stdev = 0.0; stdev <= 1.0; stdev += 0.2) {
				currFmPredictor.setStdev(stdev);

				for (double reg = 0.0; reg <= 1.0; reg += 0.2) {
					currFmPredictor.setRegulation(to_string(reg));

					for (double lr = 0.0001; lr <= 0.01; lr += 0.0005) {
						currFmPredictor.setLearningRate(lr);
						currFmPredictor.setAlgorithm("sgd");

						checkParams(trainFilename, testFilename, predFile_sgd, &currFmPredictor, bestFmPredictor, &bestResult, paramFile_sgd);
					}
				}
			}

			string line = predictionToString(bestFmPredictor, bestResult, iters);

			appendLineToFile(bestParameterFile_sgd, line);
			Logger::getInstance()->log(line, LOG_DEBUG);
		}
	}



		

	Logger::getInstance()->log("choosing parameters done! got '" + to_string(errorCount) + "' errors!", LOG_DEBUG);


}
*/




// OLD VERSION --- DONT USE THIS FUNCTION
// runs a bunch of parameter combination for all FM learning algorithms in serial
/*
TatortFMPredictor fmSerialParameterChoice(string trainFilename, string testFilename) {
	
	// run SGD with different tuning params \Theta = {standarddeviation, iterations, regulation, learnrate}
	// run ALS with different tuning params \Theta = {standarddeviation, iterations, regulation}
	// run MCMC with different tuning params \Theta = {standarddeviation, iterations}

	Logger::getInstance()->log("experimentally searching for the best model parameter choices ...", LOG_DEBUG);

	int errorCount = 0;
	
	TatortFMPredictor bestFmPredictor;
	TatortFMPredictor currFmPredictor(bestFmPredictor);


	string paramFile_sgd = "param_search_sgd.dat";
	string paramFile_als = "param_search_als.dat";
	string paramFile_mcmc = "param_search_mcmc.dat";

	string bestParameterFile = "best_param.dat";
	string bestParameterFile_mcmc = "best_param_mcmc.dat";
	string bestParameterFile_als = "best_param_als.dat";
	string bestParameterFile_sgd = "best_param_sgd.dat";

	string predFilename = "pred_result";

	writeToFile(paramFile_mcmc, "result" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
	writeToFile(paramFile_als, "result" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");
	writeToFile(paramFile_sgd, "result" + delimiter_g + "iterations" + delimiter_g + "stdev" + delimiter_g + "regulation" + delimiter_g + "learnrate\n");

	double bestResult = numeric_limits<double>::max();
	//double res;

	runFMParser(trainFilename, testFilename, DATA_UED_TENSOR);

	// 10 * 10 * 10 * 20
	// 5 * 10 * 5 * 5

	for (int iters = 20; iters <= 400; iters += 20) {
		currFmPredictor.setIterations(iters);

		for (double stdev = 0.0; stdev <= 1.0; stdev += 0.2) {
			currFmPredictor.setStdev(stdev);
			currFmPredictor.setAlgorithm("mcmc");

			checkParams(trainFilename, testFilename, predFilename, &currFmPredictor, &bestFmPredictor, &bestResult, paramFile_mcmc);

			for (double reg = 0.0; reg <= 1.0; reg += 0.2) {
				currFmPredictor.setRegulation(to_string(reg));
				currFmPredictor.setAlgorithm("als");
				
				checkParams(trainFilename, testFilename, predFilename, &currFmPredictor, &bestFmPredictor, &bestResult, paramFile_als);

				for (double lr = 0.0001; lr <= 0.01; lr += 0.0005) {
					currFmPredictor.setLearningRate(lr);
					currFmPredictor.setAlgorithm("sgd");
					
					checkParams(trainFilename, testFilename, predFilename, &currFmPredictor, &bestFmPredictor, &bestResult, paramFile_sgd);
				}
			}
		}

		string line = predictionToString(&bestFmPredictor, bestResult, iters);
		appendLineToFile(bestParameterFile, line);
	}


	Logger::getInstance()->log("choosing parameters done! got '" + to_string(errorCount) + "' errors!", LOG_DEBUG);

	

	return bestFmPredictor;

}
*/
