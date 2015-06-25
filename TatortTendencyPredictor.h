


#ifndef __TATORTTENDENCYPREDICTOR_H__
#define __TATORTTENDENCYPREDICTOR_H__



#include "Predictor.h"
#include <map>
#include <vector>


using namespace std;



class TatortTendencyPredictor : public Predictor {

protected:

	// user -> mu
	map<int, double> _userMean;

	// episode -> mu
	map<int, double> _episodeMean;

	// user -> tau
	map<int, double> _userTendencies;

	// episode -> tau
	map<int, double> _episodeTendencies;
	double _alpha;
	double _beta;

	// detective -> mu_u1, mu_u2, ...
	map<int, map<int, double> > _detectiveMeanPerUser;

	double computeMAE(vector<double> predictions, vector<double> ratings);

public:

	TatortTendencyPredictor();

	virtual vector<double> predictAllCases(vector<vector<int> > testData);
	virtual double predictCase(int userId, int episodeId, int detectiveId);

	void train(string trainFilename, string delimiter, bool isHeaderPresent);
	double test(string testFilename, string delimiter, bool isHeaderPresent, string predictionsFilename);

};







#endif
