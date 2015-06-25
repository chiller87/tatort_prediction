



#include "Parser.h"
#include "MyException.h"
#include "StringTokenizer.h"
#include "Logger.h"

#include <fstream>
#include <algorithm>
#include <sstream>
#include <string>




// ============================================================================================
// ====================================   Parser   ============================================
// ============================================================================================


Parser::Parser() {
	_numOfColumns = 0;
	_numOfLines = 0;
	_numOfDatasets = 0;
	_columns = NULL;
}

Parser::~Parser() {
	if (_columns != NULL)
		delete[] _columns;
}



void Parser::readDBFile(string filename) {

	_lines.clear();

	Logger::getInstance()->log("start reading lines from file ...", LOG_DEBUG);

	ifstream infile(filename.c_str());

	if (!infile.is_open()) {
		Logger::getInstance()->log("file '"+ filename +"'' could not be opened", LOG_ERROR);
		throw MyException("EXCEPTION: could not open file '"+ filename +"'!");
	}

	string line;
	while (getline(infile, line))
	{
		_lines.push_back(line);
	}

	_numOfLines = _lines.size();

	
	Logger::getInstance()->log("reading lines done!", LOG_DEBUG);
}




vector<string> Parser::getColumn(unsigned int index) {
	if ((int)index < _numOfColumns)
		return _columns[index];

	throw MyException("EXCEPTION: index out of bounds!");
}


int Parser::parseFile(string filename, string delimiter, bool isHeaderPresent) {
	
	// read lines of file into _lines
	readDBFile(filename);




	if (_numOfLines == 0)
		throw MyException("EXCEPTION: nothing to parse!");

	if (delimiter == "")
		throw MyException("EXCEPTION: no delimiter specified!");


	Logger::getInstance()->log("start parsing colums ...", LOG_DEBUG);
	string line = _lines[0];

	// get number of columns (first line should be seen as reference)
	vector<string> cells = StringTokenizer::justTokenize(line, delimiter, false);
	_numOfColumns = cells.size();

	// allocate memory for number of cells
	if (_columns != NULL)
		delete[] _columns;
	_columns = new vector<string>[_numOfColumns];


	int countBadLines = 0;
	
	// ignore header
	int firstDataLine;
	if(isHeaderPresent) {
		firstDataLine = 1;
		// init _headline
		_headline = cells;
	}
	else {
		firstDataLine = 0;
		_headline.clear();
	}

	_numOfDatasets = 0;


	for (int i = firstDataLine; i < _numOfLines; i++) {
		line = _lines[i];
		cells = StringTokenizer::justTokenize(line, delimiter, false);

		if ((int)cells.size() != (int)_numOfColumns) {
			countBadLines++;
			continue;
		}

		for (int j = 0; j < _numOfColumns; j++) {
			_columns[j].push_back(cells[j]);
		}
		_numOfDatasets++;

		/*
		Logger::getInstance()->log("parsed '" + line + "' to:", LOG_DEBUG);
		for (int k = 0; k < cells.size(); k++) {
			Logger::getInstance()->log("[" + to_string(k) + "] = '" + cells[k], LOG_DEBUG);
		}
		*/
	}

	Logger::getInstance()->log("parsing colums done!", LOG_DEBUG);


	return countBadLines;
}



vector<string>* Parser::getAllColumns() {
	return _columns;
}




void Parser::clear() {

	if (_columns != NULL)
		delete[] _columns;

	_columns = NULL;
	_lines.clear();
	_headline.clear();
	_numOfDatasets = 0;
	_numOfColumns = 0;
	_numOfLines = 0;
	
}





void Parser::divideLinesTrainAndTest(string sourceFilename, bool isHeaderPresent, double trainPercentage, string trainFilename, string testFilename, string predictionTargetFilename) {
	

	readDBFile(sourceFilename);

	Logger::getInstance()->log("dividing data into train ("+ std::to_string(trainPercentage) +") and test (rest) ...", LOG_DEBUG);
	
	if (_lines.size() == 0)
		throw MyException("EXCEPTION: no lines to divide!");

	
	if(isHeaderPresent)
		_numOfDatasets = _numOfLines - 1;
	else
		_numOfDatasets = _numOfLines;


	int numTrainData = _numOfDatasets * trainPercentage / 100;
	int numTestData = _numOfDatasets - numTrainData;

	// put all datasets in traindataset
	vector<int> trainData;
	for (int i = 0; i < _numOfDatasets; i++) {
		trainData.push_back(1);
	}


	int randomNumberCount = 0;
	unsigned int randomNumber;
	
	while (randomNumberCount != numTestData) {
		// pick random number in range
		randomNumber = ((double)rand() / RAND_MAX) * (_numOfDatasets - 1);
		
		// check if this dataset is already choosen
		if (trainData[randomNumber] == 1) {
			// if not, remove dataset from taindataset
			trainData[randomNumber] = 0;
			randomNumberCount++;
		}
	}



	ofstream trainFile(trainFilename.c_str());
	ofstream testFile(testFilename.c_str());
	ofstream targetFile(predictionTargetFilename);

	
	int headerOffset = 0;
	if(isHeaderPresent) {
		// write headline to files
		trainFile << _lines[0] << endl;
		testFile << _lines[0] << endl;
		headerOffset = 1;
	}



	for (int i = 0; i < _numOfDatasets; i++) {

		// if current line belongs to  train data, write it into train data file
		if (trainData[i]) {
			trainFile << _lines[i + headerOffset] << endl;
			/*if (i != (_numOfDatasets - 1)) {
				trainFile << endl;
			}*/
		}
		else {
			testFile << _lines[i + headerOffset] << endl;
			vector<string> cells = StringTokenizer::justTokenize(_lines[i + headerOffset] ,"|");
			targetFile << cells[cells.size() - 1] << endl;
			/*if (i != (_numOfDatasets - 1)) {
				testFile << endl;
			}*/
		}
	}

	Logger::getInstance()->log("dividing data ('" + trainFilename + "', '" + testFilename + "') done!", LOG_DEBUG);

}











int Parser::getNumberOfDatasets() {
	return _numOfDatasets;
}

int Parser::getNumberOfLines() {
	return _numOfLines;
}

int Parser::getNumberOfColumns() {
	return _numOfColumns;
}








