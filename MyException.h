
#ifndef __MYEXCEPTION_H__
#define __MYEXCEPTION_H__

#include <string>

class MyException {
protected:
	std::string _error;

public:
	MyException(std::string error) {
		_error = error;
	}
	std::string getErrorMsg() {
		return _error;
	}
};


#endif