

#include "TatortTendencyParser.h"
#include "Logger.h"
#include "MyException.h"

#include <fstream>
#include <algorithm>


// ============================================================================================
// =========================   TatortTendencyParser   ========================================
// ============================================================================================


TatortTendencyParser::TatortTendencyParser() : Parser() {
	_userRatingsFilename = "u_r.data";
	_episodeRatingsFilename = "e_r.data";
	_detectiveRatingsFilename = "d_r.data";
}


TatortTendencyParser::~TatortTendencyParser() {

}


void TatortTendencyParser::init() {

}





void TatortTendencyParser::addIdColumnToFile(string outfile, unsigned int columnIndex, string columnHeader, string delimiter) {



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


vector<int> TatortTendencyParser::createIds(unsigned int columnIndex) {

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



void TatortTendencyParser::parseTestData(string inFilename, string delimiter, bool isHeaderPresent) {

	_testData.clear();

	parseFile(inFilename, delimiter, isHeaderPresent);

	Logger::getInstance()->log("start parsing test data to predict ...", LOG_DEBUG);

	vector<string> userIds = getColumn(0);
	vector<string> itemIds = getColumn(2);
	vector<string> detectiveIds = getColumn(4);
	vector<string> ratings = getColumn(11);

	for (int i = 0; i < _numOfDatasets; i++) {
		vector<int> testTupel;
		testTupel.push_back(stoi(userIds[i]));
		testTupel.push_back(stoi(itemIds[i]));
		testTupel.push_back(stoi(detectiveIds[i]));

		_testData.push_back(testTupel);
		_testRatings.push_back(stod(ratings[i]));
	}

	Logger::getInstance()->log("parsing test data done!", LOG_DEBUG);
}




void TatortTendencyParser::parseTrainData(string inFilename, string delimiter, bool isHeaderPresent) {

	_userRatings.clear();
	_episodeRatings.clear();
	_detectiveRatings.clear();


	parseFile(inFilename, delimiter, isHeaderPresent);

	Logger::getInstance()->log("start parsing train data for predictor ...", LOG_DEBUG);
	vector<string> userIdColumn = getColumn(0);
	vector<string> episodeIdColumn = getColumn(2);
	vector<string> detectiveIdColumn = getColumn(4);
	vector<string> ratingColumn = getColumn(11);

	int userId;
	int episodeId;
	int detectiveId;
	double rating;

	for (int i = 0; i < _numOfDatasets; i++) {

		userId = stoi(userIdColumn[i]);
		episodeId = stoi(episodeIdColumn[i]);
		detectiveId = stoi(detectiveIdColumn[i]);
		rating = stod(ratingColumn[i]);

		// fill ratings per user
		map<int, vector<pair<int, double> > >::iterator mapIt = _userRatings.find(userId);
		if (mapIt == _userRatings.end()) {
			vector<pair<int, double> > tmp;
			tmp.push_back(pair<int, double>(episodeId, rating));
			_userRatings.insert(pair < int, vector<pair<int, double> > >(userId, tmp));
		}
		else {
			mapIt->second.push_back(pair<int, double>(episodeId, rating));
		}

		// fill ratings per episode
		mapIt = _episodeRatings.find(episodeId);
		if (mapIt == _episodeRatings.end()) {
			vector<pair<int, double> > tmp;
			tmp.push_back(pair<int, double>(userId, rating));
			_episodeRatings.insert(pair < int, vector<pair<int, double> > >(episodeId, tmp));
		}
		else {
			mapIt->second.push_back(pair<int, double>(userId, rating));
		}

		// fill ratings per detective
		mapIt = _detectiveRatings.find(detectiveId);
		if (mapIt == _detectiveRatings.end()) {
			vector<pair<int, double> > tmp;
			tmp.push_back(pair<int, double>(userId, rating));
			_detectiveRatings.insert(pair < int, vector<pair<int, double> > >(detectiveId, tmp));
		}
		else {
			mapIt->second.push_back(pair<int, double>(userId, rating));
		}
		// TODO: hier gehts weiter

	}
	Logger::getInstance()->log("parsing train data done!", LOG_DEBUG);

}




void TatortTendencyParser::writeDataToFile() {

	Logger::getInstance()->log("writing map 'userRatings' to file ...", LOG_DEBUG);
	writeMapToFile(&_userRatings, _userRatingsFilename);
	Logger::getInstance()->log("writing map 'episodeRatings' to file ...", LOG_DEBUG);
	writeMapToFile(&_episodeRatings, _episodeRatingsFilename);
	Logger::getInstance()->log("writing map 'detectiveRatings' to file ...", LOG_DEBUG);
	writeMapToFile(&_detectiveRatings, _detectiveRatingsFilename);
	Logger::getInstance()->log("writing maps to file done!", LOG_DEBUG);
}


void TatortTendencyParser::writeMapToFile(map<int, vector< pair<int, double> > > *data, string filename) {

	ofstream ofile(filename);

	if (!ofile.is_open())
		throw MyException("EXCEPTION: file '" + filename + "' could not be opened for writing");

	// DATAFILE format:
	// key
	// numberOfEntries
	// int;double
	map<int, vector< pair<int, double> > >::iterator iter = data->begin();
	while (iter != data->end()) {
		ofile << iter->first << endl;
		ofile << iter->second.size() << endl;
		for (unsigned int i = 0; i < iter->second.size(); i++) {
			ofile << iter->second[i].first << ";" << iter->second[i].second << endl;
		}
		iter++;
	}

}




void TatortTendencyParser::readDataFromFile() {


	_userRatings.clear();
	_episodeRatings.clear();
	_detectiveRatings.clear();

	Logger::getInstance()->log("reading map 'userRatings' from file ...", LOG_DEBUG);
	readMapFromFile(&_userRatings, _userRatingsFilename);
	Logger::getInstance()->log("reading map 'episodeRatings' from file ...", LOG_DEBUG);
	readMapFromFile(&_episodeRatings, _episodeRatingsFilename);
	Logger::getInstance()->log("reading map 'detectiveRatings' from file ...", LOG_DEBUG);
	readMapFromFile(&_detectiveRatings, _detectiveRatingsFilename);
	Logger::getInstance()->log("reading maps from file done!", LOG_DEBUG);

}


void TatortTendencyParser::readMapFromFile(map<int, vector< pair<int, double> > > *data, string filename) {

	ifstream ifile(filename);

	if (!ifile.is_open())
		throw MyException("EXCEPTION: file '" + filename + "' could not be opened for reading");


	// DATAFILE format:
	// key
	// numberOfEntries
	// int;double

	int keyId;
	int numberOfEntries;
	int valId;
	double val;

	while (!ifile.eof()) {
		ifile >> keyId;
		ifile >> numberOfEntries;
		vector<pair<int, double> > vec;
		char delimiter;
		for (int i = 0; i < numberOfEntries; i++) {
			ifile >> valId >> delimiter >> val;
			vec.push_back(pair<int, double>(valId, val));
		}
		data->insert(pair<int, vector< pair<int, double> > >(keyId, vec));
	}

}



void TatortTendencyParser::clear() {
	Parser::clear();

	_userRatings.clear();
	_episodeRatings.clear();
	_detectiveRatings.clear();
	_testData.clear();
}








map<int, vector< pair<int, double> > >* TatortTendencyParser::getUserRatingMap() {
	return &_userRatings;
}


map<int, vector< pair<int, double> > >* TatortTendencyParser::getEpisodeRatingMap() {
	return &_episodeRatings;
}


map<int, vector< pair<int, double> > >* TatortTendencyParser::getDetectiveRatingMap() {
	return &_detectiveRatings;
}


vector<vector<int> >* TatortTendencyParser::getTestData() {
	return &_testData;
}

vector<double>* TatortTendencyParser::getTestRatings() {
	return &_testRatings;
}







// remove every user that rated less than 'threshold' items from 'inFilename' and write cleaned data to 'outFilename'
void TatortTendencyParser::cleanData(string inFilename, string delimiter, bool isHeaderPresent, string outFilename, int threshold) {
	
	parseTrainData(inFilename, delimiter, isHeaderPresent);

	vector<int> usersToRemove;


	Logger::getInstance()->log("checking for users less than '"+ to_string(threshold) +"' ratings ...", LOG_DEBUG);
	// iterate over users and check rating count per user
	map<int, vector<pair<int, double> > >::iterator userRatingIter;
	for (userRatingIter = _userRatings.begin(); userRatingIter != _userRatings.end(); userRatingIter++) {
		int numOfRatings = userRatingIter->second.size();
		
		// if rating count is less than threshold, add user to remove list
		if (numOfRatings < threshold) {
			usersToRemove.push_back(userRatingIter->first);
		}
	}
	Logger::getInstance()->log("found '" + to_string(usersToRemove.size()) + "' users!", LOG_DEBUG);



	Logger::getInstance()->log("writing resulting data to file '"+ outFilename +"' ...", LOG_DEBUG);
	// write clean data to file
	vector<string> userIdsPerLine = getColumn(0);


	ofstream of(outFilename);

	int headerOffset = 0;
	if(isHeaderPresent) {
		// write headline to file
		of << _lines[0] << endl;
		headerOffset = 1;
	}

	

	for (int i = 0; i < _numOfDatasets; i++) {
		// if the user should not be removed, add appropriate line to file
		vector<int>::iterator userIter = find(usersToRemove.begin(), usersToRemove.end(), stoi(userIdsPerLine[i]));
		if (userIter == usersToRemove.end()) {
			of << _lines[i + headerOffset] << endl;
		}
	}

	Logger::getInstance()->log("writing clean data done!", LOG_DEBUG);
}




