
#ifndef __TATORTFMPREDICTOR_H__
#define __TATORTFMPREDICTOR_H__ 


#include <string>


#include "Predictor.h"




using namespace std;



class TatortFMPredictor : public Predictor {

protected:

	bool _useW0;
	bool _useW;
	int _numOfLatentFactors;

	string _algorithm;
	string _dim;
	int _iterations;
	double _stdev;
	double _learnRate;
	string _regulation;



public:
	TatortFMPredictor();
	~TatortFMPredictor();

	void parametersToUse(bool w0, bool w, unsigned int numOfLatentFactors);
	void setAlgorithm(string algo);
	void setIterations(unsigned int iters);
	void setRegulation(string reg);
	void setLearningRate(double lr);
	void setStdev(double stdev);

	string getAlgorithm();
	int getIterations();
	string getRegulation();
	double getLearningRate();
	double getStdev();

	virtual double trainAndTest(string trainFilename, string testFilename, string predictionFilename);

	string tuningParamsToString();

	void copyFrom(TatortFMPredictor *fmp);

};

#endif