


#include "Tools.h"
#include "Logger.h"
#include "MyException.h"

#include <vector>
#include <algorithm>
#include <random>

using namespace std;



Tools* Tools::_instance = NULL;


Tools::Tools() {
	
}

Tools::~Tools() {
	
}


Tools* Tools::getInstance() {
	if(_instance == NULL)
		_instance = new Tools();

	return _instance;
}



double Tools::computeMAE(vector<double> predictions, vector<double> ratings) {
	Logger::getInstance()->log("computing MAE ...", LOG_DEBUG);
	
	if (predictions.size() != ratings.size())
		throw MyException("EXCEPTION: size mismatch (predictions, ratings)!");


	int noPrediction = 0;
	double mae = 0;
	for (unsigned int i = 0; i < ratings.size(); i++) {
		if (predictions[i] == -1) {
			noPrediction++;
			continue;
		}
		mae += abs((ratings[i] - predictions[i]));
	}
	Logger::getInstance()->log("found no prediction for '"+ to_string(noPrediction) +"' test cases", LOG_DEBUG);

	mae = mae / ((double)ratings.size() - noPrediction);

	Logger::getInstance()->log("computing MAE done! MAE = '"+ to_string(mae) +"'", LOG_DEBUG);

	return mae;

}



unsigned int Tools::getRandomNumber(unsigned int max) {
	//return ((double)rand() / (RAND_MAX)) * (max - 1);

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<unsigned int> dist(0, max-1);

	return dist(mt);
}
