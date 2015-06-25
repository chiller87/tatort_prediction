

#include "TatortFMPredictor.h"
#include "TatortFMParser.h"
#include "Logger.h"
#include "Tools.h"

#include <sstream>





using namespace std;





TatortFMPredictor::TatortFMPredictor() {
	_algorithm = "mcmc";
	
	_iterations = 200;
	_stdev = 0.0;
	_learnRate = 0.0;
	_regulation = "0";
	

	parametersToUse(true, true, 8);
}






TatortFMPredictor::~TatortFMPredictor() {

}



void TatortFMPredictor::parametersToUse(bool w0, bool w, unsigned int numOfLatentFactors) {
	_useW0 = w0;
	_useW = w;
	_numOfLatentFactors = numOfLatentFactors;


	ostringstream os("");

	if(_useW0)
		os << "1";
	os << ",";
	if(_useW)
		os << "1";
	os << ",";
	os << _numOfLatentFactors;

	_dim = os.str();
}


double TatortFMPredictor::trainAndTest(string trainFilename, string testFilename, string predictionFilename) {

	// execute libFM command
	ostringstream libfmCmd("");

	//#ifdef linux
		libfmCmd << "./libFM ";
	//#endif
	#ifdef _WIN32
		libfmCmd << "libfm.exe ";
	#endif
	libfmCmd << "-task r ";
	libfmCmd << "-dim " << _dim << " ";
	libfmCmd << "-iter " << to_string(_iterations) << " ";
	libfmCmd << "-method " << _algorithm << " ";
	libfmCmd << "-init_stdev " << to_string(_stdev) << " ";
	libfmCmd << "-regular " << _regulation << " ";
	libfmCmd << "-learn_rate " << to_string(_learnRate) << " ";
	libfmCmd << "-train " << trainFilename << " ";
	libfmCmd << "-test " << testFilename << " ";
	libfmCmd << "-out " << predictionFilename << " ";

	Logger::getInstance()->log("executing cmd: '"+ libfmCmd.str() +"'", LOG_DEBUG);

	int ret = system(libfmCmd.str().c_str());
	
	Logger::getInstance()->log("libfm returned '"+ to_string(ret) +"'", LOG_DEBUG);


	TatortFMParser fmParser;

	vector<double> predicted = fmParser.readPredictionFromFile(predictionFilename);
	vector<double> targets = fmParser.readPredictionFromFile(testFilename);

	double mae = Tools::getInstance()->computeMAE(predicted, targets);

	return mae;
}







void TatortFMPredictor::setAlgorithm(string algo) {
	_algorithm = algo;
}

void TatortFMPredictor::setIterations(unsigned int iters) {
	_iterations = iters;
}

void TatortFMPredictor::setRegulation(string reg) {
	_regulation = reg;
}

void TatortFMPredictor::setLearningRate(double lr) {
	_learnRate = lr;
}

void TatortFMPredictor::setStdev(double stdev) {
	_stdev = stdev;
}


string TatortFMPredictor::getAlgorithm() {
	return _algorithm;
}

int TatortFMPredictor::getIterations() {
	return _iterations;
}

string TatortFMPredictor::getRegulation() {
	return _regulation;
}

double TatortFMPredictor::getLearningRate() {
	return _learnRate;
}

double TatortFMPredictor::getStdev() {
	return _stdev;
}


string TatortFMPredictor::tuningParamsToString() {
	ostringstream os("");

	os << "algorithm|stdev|iterations|regulation|learnrate" << endl;
	os << _algorithm << "|";
	os << _stdev << "|";
	os << _iterations << "|";
	os << _regulation << "|";
	os << _learnRate;

	return os.str();
}

void TatortFMPredictor::copyFrom(TatortFMPredictor *fmp) {
	this->setAlgorithm(fmp->getAlgorithm());
	this->setStdev(fmp->getStdev());
	this->setIterations(fmp->getIterations());
	this->setRegulation(fmp->getRegulation());
	this->setLearningRate(fmp->getLearningRate());
}

