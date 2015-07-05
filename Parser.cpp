



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
		//Logger::getInstance()->log(line, LOG_DEBUG);
		_lines.push_back(line);
	}

	_numOfLines = _lines.size();

	
	Logger::getInstance()->log("reading lines done!", LOG_DEBUG);
}



void Parser::addMissingViewersAndQuotes(string filename, string delimiter, int viewerIndex, int quoteIndex) {


	Logger::getInstance()->log("start completing viewer and quote column ...", LOG_DEBUG);

	parseFile(filename, delimiter, true);

	vector<string> vec;

	string s("|NULL|");
	vec.push_back(s.substr(1, 4));
	Logger::getInstance()->log("created string: "+vec[0], LOG_DEBUG);

	Logger::getInstance()->log("computing means ...", LOG_DEBUG);
	double viewerMean = 0.0;
	double quoteMean = 0.0;
	int numViewerValues = 0;
	int numQuoteValues = 0;
	for (int i = 0; i < _numOfDatasets; i++) {
		//Logger::getInstance()->log(to_string(i), LOG_DEBUG);
		//Logger::getInstance()->log(_lines[i+1], LOG_DEBUG);
		//Logger::getInstance()->log(_columns[viewerIndex][i], LOG_DEBUG);
		if(_columns[viewerIndex][i] != "NULL") {
			//Logger::getInstance()->log(_columns[viewerIndex][i], LOG_DEBUG);
			viewerMean += stod(_columns[viewerIndex][i]);
			numViewerValues++;
		}

		if(_columns[quoteIndex][i] != "NULL") {
			replace(_columns[quoteIndex][i].begin(), _columns[quoteIndex][i].end(), ',', '.');
			quoteMean += stod(_columns[quoteIndex][i]);
			numQuoteValues++;
		}
	}

	viewerMean = viewerMean / numViewerValues;
	quoteMean = quoteMean / numQuoteValues;

	Logger::getInstance()->log("updateing columns ...", LOG_DEBUG);
	int numViewerModified = 0;
	int numQuoteModified = 0;
	for (int i = 0; i < _numOfDatasets; i++) {
		if(_columns[viewerIndex][i] == "NULL") {
			_columns[viewerIndex][i] = to_string(viewerMean);
			numViewerModified++;
		}

		if(_columns[quoteIndex][i] == "NULL") {
			_columns[quoteIndex][i] = to_string(quoteMean);
			numQuoteModified++;
		}
	}

	Logger::getInstance()->log("updated '"+to_string(numViewerModified)+"' viewers and '"+to_string(numQuoteModified)+"' quotes.", LOG_DEBUG);


	
	ofstream of(filename);
	if (!of.is_open()) {
		Logger::getInstance()->log("cannot open file '"+filename+"'!", LOG_CRITICAL);
		throw MyException("EXCEPTION: cannot open file!");
	}

	Logger::getInstance()->log("writing to file ...", LOG_DEBUG);
	if((int)_headline.size() == _numOfColumns) {
		for (unsigned int i = 0; i < _headline.size(); i++) {

			if(i != 0)
				of << delimiter;

			of << _headline[i];
		}
		of << endl;
	}

	// add contents to output file
	for (int i = 0; i < _numOfDatasets; i++) {

		for (int j = 0; j < _numOfColumns; j++) {
			// no delimiter at beginning
			if (j != 0)
				of << delimiter;

			of << _columns[j][i];
		}

		if (i < (_numOfDatasets - 1))
			of << endl;
	}
	


	Logger::getInstance()->log("completing viewer and quote column done!", LOG_DEBUG);

}



void Parser::addIdColumnToFile(string outfile, unsigned int columnIndex, string columnHeader, string delimiter) {



	// create UserIDs from names in source file
	vector<int> idColumn = createIds(columnIndex);

	Logger::getInstance()->log("writing user IDs and other columns to file '" + outfile + "'...", LOG_DEBUG);
	ofstream of(outfile);
	if (!of.is_open()) {
		Logger::getInstance()->log("cannot open file!", LOG_CRITICAL);
		throw MyException("EXCEPTION: cannot open file!");
	}



	// if headline was parsed, add it to the output file by adding appropriate 'ID'
	if ((int)_headline.size() == _numOfColumns) {
		for (unsigned int i = 0; i < _headline.size(); i++) {

			// no delimiter at beginning
			if (i != 0)
				of << delimiter;

			// put the ID header in
			if (i == columnIndex)
				of << columnHeader << delimiter;

			of << _headline[i];
		}

		of << endl;
	}


	// add contents to output file by adding new 'ID' column
	for (int i = 0; i < _numOfDatasets; i++) {

		for (int j = 0; j < _numOfColumns; j++) {
			// no delimiter at beginning
			if (j != 0)
				of << delimiter;

			// put 
			if (j == (int)columnIndex)
				of << idColumn[i] << delimiter;

			of << _columns[j][i];
		}

		if (i < (_numOfDatasets - 1))
			of << endl;

	}

	Logger::getInstance()->log("writing done!", LOG_DEBUG);

}


vector<int> Parser::createIds(unsigned int columnIndex) {

	Logger::getInstance()->log("start creating user IDs ...", LOG_DEBUG);

	if ((int)columnIndex > _numOfColumns)
		throw MyException("EXCEPTION: index out of bounds!");

	vector<string> dataColumn = getColumn(columnIndex);

	vector<string> data;

	vector<int> idColumn;

	int id;
	for (unsigned int i = 0; i < dataColumn.size(); i++) {
		string name = dataColumn[i];

		// transform name to lower case to avoid kind of 'overfitting'
		transform(name.begin(), name.end(), name.begin(), ::tolower);

		// search current name
		vector<string>::iterator it = find(data.begin(), data.end(), name);

		// user not already recognized
		if (it == data.end()) {
			// add name to data
			data.push_back(name);

			// remember new id
			id = data.size() - 1;
		}
		else {
			// the id is the difference from 'it' to 'data.begin()'
			id = it - data.begin();
		}

		// add id (start by 1)
		idColumn.push_back(id + 1);
	}

	if (idColumn.size() != dataColumn.size())
		throw MyException("EXCEPTION: column size missmatch");

	Logger::getInstance()->log("creating user IDs done!", LOG_DEBUG);
	return idColumn;
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
		randomNumber = ((double)rand() / (RAND_MAX)) * (_numOfDatasets - 1);
		
		// check if this dataset is already choosen
		if (trainData[randomNumber] == 1) {
			// if not, remove dataset from taindataset
			trainData[randomNumber] = 0;
			randomNumberCount++;
		}
	}


	/*
	ofstream outfile("chosen_train_data.dat");
	for(int i = 0; i < trainData.size(); i++) {
		outfile << trainData[i] << endl;
	}
	*/


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








