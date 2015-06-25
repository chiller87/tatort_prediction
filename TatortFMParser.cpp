



#include "TatortFMParser.h"
#include "MyException.h"
#include "Logger.h"


#include <fstream>
#include <vector>
#include <algorithm>


using namespace std;


TatortFMParser::TatortFMParser() : Parser() {

}

TatortFMParser::~TatortFMParser() {
	Parser::~Parser();
}


void TatortFMParser::convertDataToTensor(string inTrainFilename, string inTestFilename, string delimiter, string outTrainFilename, string outTestFilename, bool isHeaderPresent) {

	Parser trainParser;
	Parser testParser;

	trainParser.parseFile(inTrainFilename, delimiter, isHeaderPresent);
	testParser.parseFile(inTestFilename, delimiter, isHeaderPresent);


	Logger::getInstance()->log("start converting data to libFM conform format ...", LOG_DEBUG);

	int numTrainData = trainParser.getNumberOfDatasets();
	int numTestData = testParser.getNumberOfDatasets();

	Logger::getInstance()->log("number of train datasets '"+ to_string(numTrainData) +"'", LOG_DEBUG);
	Logger::getInstance()->log("number of test datasets '"+ to_string(numTestData) +"'", LOG_DEBUG);

	if(trainParser.getNumberOfColumns() != testParser.getNumberOfColumns())
		throw MyException("EXCEPTION: number of columns mismatch between train and test data!");

	if((numTrainData <= 0) || (numTestData <= 0))
		throw MyException("EXCEPTION: no datasets to convert!");

	int numOfColumns = trainParser.getNumberOfColumns();

	Logger::getInstance()->log("number of columns '" + to_string(numOfColumns) + "'", LOG_DEBUG);

	vector<string> trainRatings = trainParser.getColumn(numOfColumns - 1);
	vector<string> trainUserIds = trainParser.getColumn(0);
	vector<string> trainEpisodeIds = trainParser.getColumn(2);
	vector<string> trainDetectiveIds = trainParser.getColumn(4);

	Logger::getInstance()->log("train data initialized", LOG_DEBUG);

	vector<string> testRatings = testParser.getColumn(numOfColumns - 1);
	vector<string> testUserIds = testParser.getColumn(0);
	vector<string> testEpisodeIds = testParser.getColumn(2);
	vector<string> testDetectiveIds = testParser.getColumn(4);

	Logger::getInstance()->log("test data initialized", LOG_DEBUG);

	int maxUserId = max(stoi(trainUserIds[0]), stoi(testUserIds[0]));
	int maxEpisodeId = max(stoi(trainEpisodeIds[0]), stoi(testEpisodeIds[0]));
	int maxDetectiveId = max(stoi(trainDetectiveIds[0]), stoi(trainDetectiveIds[0]));

	Logger::getInstance()->log("max variables initialized", LOG_DEBUG);

	int maxNumDatasets = max(numTestData, numTrainData);

	

	// looking for max Ids
	for(int i = 1; i < maxNumDatasets; i++) {
		
		//Logger::getInstance()->log("iteration '"+ to_string(i) +"'", LOG_DEBUG);

		if(i < numTrainData) {
			maxUserId = max(maxUserId, stoi(trainUserIds[i]));
			maxEpisodeId = max(maxEpisodeId, stoi(trainEpisodeIds[i]));
			maxDetectiveId = max(maxDetectiveId, stoi(trainDetectiveIds[i]));
		}

		//Logger::getInstance()->log("train done!", LOG_DEBUG);

		if(i < numTestData) {
			//Logger::getInstance()->log("numTestData = '"+ to_string(numTestData) +"' and i = '"+ to_string(i) +"'", LOG_DEBUG);
			maxUserId = max(maxUserId, stoi(testUserIds[i]));
			maxEpisodeId = max(maxEpisodeId, stoi(testEpisodeIds[i]));
			maxDetectiveId = max(maxDetectiveId, stoi(testDetectiveIds[i]));
		}		
	}


	Logger::getInstance()->log("max ids computed", LOG_DEBUG);




	ofstream ofTrain(outTrainFilename);
	if(!ofTrain.is_open())
		throw MyException("EXCEPTION: could not open file '"+ outTrainFilename +"'!");


	ofstream ofTest(outTestFilename);
	if(!ofTest.is_open())
		throw MyException("EXCEPTION: could not open file '"+ outTestFilename +"'!");

	Logger::getInstance()->log("output files successfully opened", LOG_DEBUG);


	// writing data in libfm conform format to file
	for (int i = 0; i < maxNumDatasets; i++)
	{
		int userOffset = 0;
		int episodeOffset = maxUserId;
		int detectiveOffset = episodeOffset + maxEpisodeId;

		if(i < numTrainData) {
			ofTrain << stod(trainRatings[i]) << " ";
			ofTrain << (userOffset + stoi(trainUserIds[i])) << ":1 ";
			ofTrain << (episodeOffset + stoi(trainEpisodeIds[i])) << ":1 ";
			ofTrain << (detectiveOffset + stoi(trainDetectiveIds[i])) << ":1 ";
			ofTrain << endl;
		}

		if(i < numTestData) {
			ofTest << stod(testRatings[i]) << " ";
			ofTest << (userOffset + stoi(testUserIds[i])) << ":1 ";
			ofTest << (episodeOffset + stoi(testEpisodeIds[i])) << ":1 ";
			ofTest << (detectiveOffset + stoi(testDetectiveIds[i])) << ":1 ";
			ofTest << endl;
		}
	}

	Logger::getInstance()->log("converting data to libFM conform format done!", LOG_DEBUG);



}





void TatortFMParser::convertDataToMatrix(string inTrainFilename, string inTestFilename, string delimiter, string outTrainFilename, string outTestFilename, bool isHeaderPresent) {

	Parser trainParser;
	Parser testParser;

	trainParser.parseFile(inTrainFilename, delimiter, isHeaderPresent);
	testParser.parseFile(inTestFilename, delimiter, isHeaderPresent);


	Logger::getInstance()->log("start converting data to libFM conform format ...", LOG_DEBUG);

	int numTrainData = trainParser.getNumberOfDatasets();
	int numTestData = testParser.getNumberOfDatasets();

	Logger::getInstance()->log("number of train datasets '"+ to_string(numTrainData) +"'", LOG_DEBUG);
	Logger::getInstance()->log("number of test datasets '"+ to_string(numTestData) +"'", LOG_DEBUG);

	if(trainParser.getNumberOfColumns() != testParser.getNumberOfColumns())
		throw MyException("EXCEPTION: number of columns mismatch between train and test data!");

	if((numTrainData <= 0) || (numTestData <= 0))
		throw MyException("EXCEPTION: no datasets to convert!");

	int numOfColumns = trainParser.getNumberOfColumns();

	vector<string> trainRatings = trainParser.getColumn(numOfColumns - 1);
	vector<string> trainUserIds = trainParser.getColumn(0);
	vector<string> trainEpisodeIds = trainParser.getColumn(2);

	vector<string> testRatings = testParser.getColumn(numOfColumns - 1);
	vector<string> testUserIds = testParser.getColumn(0);
	vector<string> testEpisodeIds = testParser.getColumn(2);

	int maxUserId = max(stoi(trainUserIds[0]), stoi(testUserIds[0]));
	int maxEpisodeId = max(stoi(trainEpisodeIds[0]), stoi(testEpisodeIds[0]));


	int maxNumDatasets = max(numTestData, numTrainData);

	

	// looking for max Ids
	for(int i = 1; i < maxNumDatasets; i++) {
		
		//Logger::getInstance()->log("iteration '"+ to_string(i) +"'", LOG_DEBUG);

		if(i < numTrainData) {
			maxUserId = max(maxUserId, stoi(trainUserIds[i]));
			maxEpisodeId = max(maxEpisodeId, stoi(trainEpisodeIds[i]));
		}

		//Logger::getInstance()->log("train done!", LOG_DEBUG);

		if(i < numTestData) {
			//Logger::getInstance()->log("numTestData = '"+ to_string(numTestData) +"' and i = '"+ to_string(i) +"'", LOG_DEBUG);
			maxUserId = max(maxUserId, stoi(testUserIds[i]));
			maxEpisodeId = max(maxEpisodeId, stoi(testEpisodeIds[i]));
		}

		
	}


	ofstream ofTrain(outTrainFilename);
	if(!ofTrain.is_open())
		throw MyException("EXCEPTION: could not open file '"+ outTrainFilename +"'!");


	ofstream ofTest(outTestFilename);
	if(!ofTest.is_open())
		throw MyException("EXCEPTION: could not open file '"+ outTestFilename +"'!");


	// writing data in libfm conform format to file
	for (int i = 0; i < maxNumDatasets; i++)
	{
		int userOffset = 0;
		int episodeOffset = maxUserId;

		if(i < numTrainData) {
			ofTrain << stod(trainRatings[i]) << " ";
			ofTrain << (userOffset + stoi(trainUserIds[i])) << ":1 ";
			ofTrain << (episodeOffset + stoi(trainEpisodeIds[i])) << ":1 ";
			ofTrain << endl;
		}

		if(i < numTestData) {
			ofTest << stod(testRatings[i]) << " ";
			ofTest << (userOffset + stoi(testUserIds[i])) << ":1 ";
			ofTest << (episodeOffset + stoi(testEpisodeIds[i])) << ":1 ";
			ofTest << endl;
		}
	}

	Logger::getInstance()->log("converting data to libFM conform format done!", LOG_DEBUG);



}




vector<double> TatortFMParser::readPredictionFromFile(string predictionFilename) {

	Logger::getInstance()->log("start reading prediction from file '"+ predictionFilename +"' ...", LOG_DEBUG);

	ifstream infile(predictionFilename.c_str());

	if (!infile.is_open()) {
		Logger::getInstance()->log("file '"+ predictionFilename +"'' could not be opened", LOG_ERROR);
		throw MyException("could not open file '"+ predictionFilename +"'!");
	}

	vector<double> result;

	string line;
	while (infile.good())
	{
		double val;
		infile >> val;
		result.push_back(val);
		getline(infile, line);
	}

	Logger::getInstance()->log("reading prediction done!", LOG_DEBUG);

	return result;

}

vector<double> TatortFMParser::readTargetsFromFile(string testFilename) {
		
	Logger::getInstance()->log("start reading targets from file '"+ testFilename +"' ...", LOG_DEBUG);

	parseFile(testFilename, " ", false);

	vector <string> strRes = getColumn(_numOfColumns - 1);

	vector<double> dRes(strRes.size());

	Logger::getInstance()->log("converting targets to double ... ", LOG_DEBUG);
	for(unsigned int i = 0; i < strRes.size(); i++) {
		Logger::getInstance()->log("converting target '"+ strRes[i] +"' ... ", LOG_DEBUG);
		dRes[i] = stod(strRes[i]);
	}

	Logger::getInstance()->log("reading and converting targets done!", LOG_DEBUG);

	return dRes;
}




