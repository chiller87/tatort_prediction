



#include "TatortTendencyParser.h"
#include "TatortTendencyPredictor.h"
#include "MyException.h"
#include "Logger.h"
#include "Tools.h"

#include <fstream>
#include <algorithm>



TatortTendencyPredictor::TatortTendencyPredictor() {
	_alpha = 0.4;
	_beta = 0.4;
}



void TatortTendencyPredictor::train(string trainFilename, string delimiter, bool isHeaderPresent) {
	
	
	
	TatortTendencyParser tp;

	try {
		tp.parseTrainData(trainFilename, delimiter, isHeaderPresent);
	}
	catch (MyException e) {
		Logger::getInstance()->log(e.getErrorMsg(), LOG_CRITICAL);
		return;
	}

	Logger::getInstance()->log("start training ...", LOG_DEBUG);

	map<int, vector< pair<int, double> > >* userRatings = tp.getUserRatingMap();
	map<int, vector< pair<int, double> > >* episodeRatings = tp.getEpisodeRatingMap();
	map<int, vector< pair<int, double> > >* detectiveRatings = tp.getDetectiveRatingMap();

	//int userCount = 0;
	//int episodeCount = 0;
	
	// compute user means
	Logger::getInstance()->log("computing user means ...", LOG_DEBUG);
	map<int, vector< pair<int, double> > >::iterator userRatingIter;
	for (userRatingIter = userRatings->begin(); userRatingIter != userRatings->end(); userRatingIter++) {
		double mu = 0;
		int ratingCount = 0;
		vector< pair<int, double> >::iterator ratingIter;
		for (ratingIter = userRatingIter->second.begin(); ratingIter != userRatingIter->second.end(); ratingIter++) {
			mu += ratingIter->second;
			ratingCount++;
		}

		mu = mu / (double)ratingCount;
		_userMean.insert(pair<int, double>(userRatingIter->first, mu));
	}


	// compute episode means
	map<int, vector< pair<int, double> > >::iterator episodeRatingIter;
	Logger::getInstance()->log("computing episode means ...", LOG_DEBUG);
	for (episodeRatingIter = episodeRatings->begin(); episodeRatingIter != episodeRatings->end(); episodeRatingIter++) {
		double mu = 0;
		int ratingCount = 0;
		vector< pair<int, double> >::iterator ratingIter;
		for (ratingIter = episodeRatingIter->second.begin(); ratingIter != episodeRatingIter->second.end(); ratingIter++) {
			mu += ratingIter->second;
			ratingCount++;
		}

		mu = mu / (double)ratingCount;
		_episodeMean.insert(pair<int, double>(episodeRatingIter->first, mu));
	}



	// compute user tendencies
	Logger::getInstance()->log("computing user tendencies ...", LOG_DEBUG);
	for (userRatingIter = userRatings->begin(); userRatingIter != userRatings->end(); userRatingIter++) {
		double tau = 0;
		vector< pair<int, double> >::iterator ratingIter;
		map<int, double >::iterator episodeMeanIter;
		for (ratingIter = userRatingIter->second.begin(); ratingIter != userRatingIter->second.end(); ratingIter++) {
			episodeMeanIter = _episodeMean.find(ratingIter->first);
			if (episodeMeanIter == _episodeMean.end())
				throw MyException("EXCEPTION: episode not found!");

			tau += (ratingIter->second - episodeMeanIter->second);
		}
		tau = tau / userRatingIter->second.size();
		_userTendencies.insert(pair<int, double>(userRatingIter->first, tau));
	}


	// compute episode tendencies
	Logger::getInstance()->log("computing episode tendencies ...", LOG_DEBUG);
	for (episodeRatingIter = episodeRatings->begin(); episodeRatingIter != episodeRatings->end(); episodeRatingIter++) {
		double tau = 0;
		vector< pair<int, double> >::iterator ratingIter;
		map<int, double >::iterator userMeanIter;

		for (ratingIter = episodeRatingIter->second.begin(); ratingIter != episodeRatingIter->second.end(); ratingIter++) {
			userMeanIter = _userMean.find(ratingIter->first);
			if (userMeanIter == _userMean.end())
				throw MyException("EXCEPTION: episode not found!");

			tau += (ratingIter->second - userMeanIter->second);
		}
		tau = tau / episodeRatingIter->second.size();
		_episodeTendencies.insert(pair<int, double>(episodeRatingIter->first, tau));
	}



	// compute detective mean per user
	Logger::getInstance()->log("computing detective mean per user ...", LOG_DEBUG);

	// iterate over all detectives
	map<int, vector< pair<int, double> > >::iterator detectiveRatingIter;
	for (detectiveRatingIter = detectiveRatings->begin(); detectiveRatingIter != detectiveRatings->end(); detectiveRatingIter++) {
		map<int, int> numberOfRatingsByUser;

		vector< pair<int, double> >::iterator ratingIter;
		int detectiveId = detectiveRatingIter->first;

		// iterate over all user ratings for current detective
		for (ratingIter = detectiveRatingIter->second.begin(); ratingIter != detectiveRatingIter->second.end(); ratingIter++) {
			int userId = ratingIter->first;
			double rating = ratingIter->second;

			// check if detective is already in destination map (_detectiveMeanPerUser)
			map<int, map<int, double> >::iterator detectiveMeanIter = _detectiveMeanPerUser.find(detectiveId);
			
			if (detectiveMeanIter == _detectiveMeanPerUser.end()) {
				// if not, add it with new entry
				map<int, double> tmp;
				tmp.insert(pair<int, double>(userId, rating));
				_detectiveMeanPerUser.insert(pair<int, map<int, double> >(detectiveId, tmp));
				numberOfRatingsByUser.insert(pair<int, int>(userId, 1));
			}
			else {
				// if it is in, check if the current user is in
				map<int, double>::iterator userSumIterator = detectiveMeanIter->second.find(userId);
				if (userSumIterator == detectiveMeanIter->second.end()) {
					// if not, add it with its rating
					detectiveMeanIter->second.insert(pair<int, double>(userId, rating));
					numberOfRatingsByUser.insert(pair<int, int>(userId, 1));
				}
				else {
					// if it is, add the rating to appropriate user
					userSumIterator->second += rating;
					numberOfRatingsByUser.at(userId) += 1;
				}
			}
		}


		// divide each sum of user ratings by appropriate rating count of user
		map<int, double>::iterator userMeanIter = _detectiveMeanPerUser.find(detectiveId)->second.begin();
		for (; userMeanIter != _detectiveMeanPerUser.find(detectiveId)->second.end(); userMeanIter++) {
			userMeanIter->second /= numberOfRatingsByUser.at(userMeanIter->first);
		}

	}



	Logger::getInstance()->log("training done!", LOG_DEBUG);

	


}









double TatortTendencyPredictor::test(string testFilename, string delimiter, bool isHeaderPresent, string predictionsFilename) {
	TatortTendencyParser tp;


	// parse test data	
	tp.parseTestData(testFilename, delimiter, isHeaderPresent);

	
	
	vector<vector<int> > *testData = tp.getTestData();
	vector<double> *ratings = tp.getTestRatings();

	vector<double> predictions = predictAllCases(*testData);
	

	ofstream of(predictionsFilename);

	if(!of.is_open())
		throw MyException("could not open file '"+ predictionsFilename +"'!");

	for(unsigned int i = 0; i < predictions.size(); i++) {
		of << to_string(predictions[i]) << endl;
	}


	double error = Tools::getInstance()->computeMAE(predictions, *ratings);



	return error;

}








vector<double> TatortTendencyPredictor::predictAllCases(vector<vector<int> > testData) {
	
	
	Logger::getInstance()->log("predicting all cases ...", LOG_DEBUG);
	vector<double> prediction;

	for (unsigned int sample = 0; sample < testData.size(); sample++) {
		if (testData[sample].size() != 3) {
			prediction.push_back(-1);
			continue;
		}

		try {
			double rating = predictCase(testData[sample][0], testData[sample][1], testData[sample][2]);
			prediction.push_back(rating);
		}
		catch (MyException e) {
			Logger::getInstance()->log("ignoring prediction, because of: "+ e.getErrorMsg() , LOG_DEBUG);
			prediction.push_back(-1);
		}
	}

	Logger::getInstance()->log("prediction done!", LOG_DEBUG);

	return prediction;
}



double TatortTendencyPredictor::predictCase(int userId, int episodeId, int detectiveId) {
	

	bool isUserTendencyPresent = true;
	bool isEpisodeTendencyPresent = true;
	bool isUserMeanPresent = true;
	bool isEpisodeMeanPresent = true;
	bool isDetectiveUserMeanPresent = true;

	double tau_u = 0.0, tau_i = 0.0, mu_i = 0.0, mu_u = 0.0, mu_ud = 0.0;
	// get user tendency for user u
	map<int, double>::iterator iter = _userTendencies.find(userId);
	if (iter == _userTendencies.end()) {
		isUserTendencyPresent = false;
	}
	else {
		tau_u = iter->second;
	}
	// get episode tendency for episode i
	iter = _episodeTendencies.find(episodeId);
	if (iter == _episodeTendencies.end()) {
		isEpisodeTendencyPresent = false;
	}
	else {
		tau_i = iter->second;
	}
	

	// get episode mean for episode i
	iter = _episodeMean.find(episodeId);
	if (iter == _episodeMean.end()) {
		isEpisodeMeanPresent = false;
	}
	else {
		mu_i = iter->second;
	}

	// get user mean for user u
	iter = _userMean.find(userId);
	if (iter == _userMean.end()) {
		isUserMeanPresent = false;
	}
	else{
		mu_u = iter->second;
	}

	// get user mean of detective d for user u
	map<int, map<int, double> >::iterator detectiveUserIter = _detectiveMeanPerUser.find(detectiveId);
	if (detectiveUserIter == _detectiveMeanPerUser.end())
		throw MyException("EXCEPTION: detectiveId (d='" + to_string(detectiveId) + "') out of bounds!");
	

	
	// if the user did not rate any episode with the current detective, just take the tendency rating
	map<int, double>::iterator userIter = detectiveUserIter->second.find(userId);
	if (userIter == detectiveUserIter->second.end()) {
		//throw MyException("EXCEPTION: no user_mean for current detectiveId (d='" + to_string(detectiveId) +"', u='" + to_string(userId) + "')!");
		isDetectiveUserMeanPresent = false;
	}
	else {
		mu_ud = userIter->second;
	}


	if(!isUserTendencyPresent) {
		if(isEpisodeMeanPresent) {
			Logger::getInstance()->log("userId (u='" + to_string(userId) + "') out of bounds, taking item mean as prediction!", LOG_INFO);
			return mu_i;
		}
		else {
			throw MyException("EXCEPTION: userId (u='" + to_string(userId) + "') out of bounds!");
		}
	}
	if(!isUserMeanPresent) {
		if(isEpisodeMeanPresent) {
			Logger::getInstance()->log("userId (u='" + to_string(userId) + "') out of bounds, taking item mean as prediction!", LOG_INFO);
			return mu_i;
		}
		else {
			throw MyException("EXCEPTION: userId (u='" + to_string(userId) + "') out of bounds!");
		}
	}
	if(!isEpisodeTendencyPresent) {
		throw MyException("EXCEPTION: episodeId (i='" + to_string(episodeId) + "') out of bounds!");
	}
	if(!isEpisodeMeanPresent) {
		throw MyException("EXCEPTION: episodeId (i='" + to_string(episodeId) + "') out of bounds!");
	}
	

	double rating = 0.0;

	// compute rating
	if ((tau_u >= 0) && (tau_i >= 0)) {
		rating = max((mu_u + tau_u), (mu_i + tau_i));
	}
	else if ((tau_u < 0) && (tau_i < 0)) {
		rating = min((mu_u + tau_u), (mu_i + tau_i));
	}
	else if ((tau_u < 0) && (tau_i >= 0)) {
		rating = min( max(mu_u, (((mu_i + tau_u) * _beta) + (mu_u + tau_i) * (1 - _beta)) ), mu_i );
	}
	else if ((tau_u >= 0) && (tau_i < 0)) {
		rating = mu_i * _beta + mu_u * (1 - _beta);
	}

	// if user-detective mean is present ...
	if (isDetectiveUserMeanPresent) {
		// add detective
		rating = (_alpha * mu_ud) + ((1 - _alpha) * rating);
	}

	return rating;
}




