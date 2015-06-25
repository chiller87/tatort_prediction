


#include "Logger.h"
#include <iostream>



Logger *Logger::_instance = NULL;



Logger::Logger() {
	_currVerbosityLevel = LOG_ERROR;
}




Logger* Logger::getInstance() {
	if (Logger::_instance == NULL) {
		Logger::_instance = new Logger();
	}

	return Logger::_instance;
}








void Logger::setVerbosityLevel(int level) {
	if (level >= LOG_NOTHING && level <= LOG_DEBUG)
		_currVerbosityLevel = level;
}


void Logger::log(std::string msg, int level) {
	
	if (level > _currVerbosityLevel)
		return;


	
	switch (level) {
	case LOG_CRITICAL:
		std::cout << "CRITICAL: " << msg << std::endl;
		break;
	case LOG_ERROR:
		std::cout << "ERROR: " << msg << std::endl;
		break;
	case LOG_WARNING:
		std::cout << "WARNING: " << msg << std::endl;
		break;
	case LOG_INFO:
		std::cout << "INFO: " << msg << std::endl;
		break;
	case LOG_DEBUG:
		std::cout << "DEBUG: " << msg << std::endl;
		break;
	}
}





