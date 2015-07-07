#ifndef __TOOLS_H__
#define __TOOLS_H__ 



#include <vector>





class Tools {
private:
	Tools();
	Tools(Tools& t);

	static Tools *_instance;

public:
	~Tools();

	static Tools* getInstance();

	double computeMAE(std::vector<double> predictions, std::vector<double> ratings);
	unsigned int getRandomNumber(unsigned int max);


};


#endif