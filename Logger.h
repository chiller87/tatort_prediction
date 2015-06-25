



#ifndef __LOGGER_H__
#define __LOGGER_H__


#include <string>


#define LOG_NOTHING 0
#define LOG_CRITICAL 1
#define LOG_ERROR 2
#define LOG_WARNING 3
#define LOG_INFO 4
#define LOG_DEBUG 5




class Logger {
protected:
	static Logger *_instance;
	int _currVerbosityLevel;

	Logger();
public:
	
	static Logger* getInstance();
	void setVerbosityLevel(int level);
	void log(std::string msg, int level);

};


#endif




