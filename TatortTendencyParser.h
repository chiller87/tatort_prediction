#ifndef __TATORTTENDENCYPARSER_H__
#define __TATORTTENDENCYPARSER_H__


#include <map>
#include <vector>


#include "Parser.h"


class TatortTendencyParser : public Parser {

protected:
	//MEMBER
	// user -> (episode, rating)
	map<int, vector< pair<int, double> > > _userRatings;
	string _userRatingsFilename;
	// episode -> (user, rating)
	map<int, vector< pair<int, double> > > _episodeRatings;
	string _episodeRatingsFilename;
	// detective -> (user, rating)
	map<int, vector< pair<int, double> > > _detectiveRatings;
	string _detectiveRatingsFilename;

	// userId_1 itemId_1 detectiveId_1 rating_1
	// userId_2 itemId_2 detectiveId_2 rating_2
	// ...
	vector<vector<int> > _testData;

	vector<double> _testRatings;


	vector<int> createUserIds(unsigned int userColumnIndex);
	vector<int> createIds(unsigned int columnIndex);



public:
	TatortTendencyParser();
	~TatortTendencyParser();

	void init();

	virtual void parseTrainData(string inFilename, string delimiter, bool isHeaderPresent);
	virtual void parseTestData(string inFilename, string delimiter, bool isHeaderPresent);
	virtual void writeDataToFile();
	virtual void readDataFromFile();
	vector<int> parseUsers();
	void addUserIdToFile(string outfile, unsigned int userColumnIndex);
	void addIdColumnToFile(string outfile, unsigned int columnIndex, string columnHeader, string delimiter);

	void writeMapToFile(map<int, vector< pair<int, double> > > *data, string filename);
	void readMapFromFile(map<int, vector< pair<int, double> > > *data, string filename);
	virtual void clear();

	void cleanData(string inFilename, string delimiter, bool isHeaderPresent, string outFilename, int threshold);

	map<int, vector< pair<int, double> > >* getUserRatingMap();
	map<int, vector< pair<int, double> > >* getEpisodeRatingMap();
	map<int, vector< pair<int, double> > >* getDetectiveRatingMap();
	vector<vector<int> >* getTestData();
	vector<double>* getTestRatings();
};




#endif