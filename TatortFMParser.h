

#ifndef __TATORTFMPARSER_H__
#define __TATORTFMPARSER_H__


#include "Parser.h"

#include <string>


using namespace std;


class TatortFMParser : public Parser {

protected:


public:
	TatortFMParser();
	virtual ~TatortFMParser();

	// parses train and test data and writes them in libFM conform file format.
	void convertDataToMatrix(string inTrainFilename, string inTestFilename, string delimiter, string outTrainFilename, string outTestFilename, bool isHeaderPresent);
	void convertDataToTensor(string inTrainFilename, string inTestFilename, string delimiter, string outTrainFilename, string outTestFilename, bool isHeaderPresent);
	void convertDataToTensorPlusAttributes(string inTrainFilename, string inTestFilename, string delimiter, vector<unsigned int> attributeIndices, string outTrainFilename, string outTestFilename, bool isHeaderPresent);

	vector<double> readPredictionFromFile(string predictionFilename);
	vector<double> readTargetsFromFile(string testFilename);
	

};



#endif
