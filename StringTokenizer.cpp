

#include "StringTokenizer.h"
#include "MyException.h"
#include <iostream>
#include <iomanip>

using namespace std;


StringTokenizer::StringTokenizer(string source, string delimiter, bool ignoreEmptyTokens) {
	

	_source = source;
	_delimiter = delimiter;
	_ignoreEmptyTokens = ignoreEmptyTokens;

	_count = 0;


}



void StringTokenizer::tokenize() {
	int start = 0;
	int stop = 0 - _delimiter.size();
	int length;

	bool empty = false;

	_tokens.clear();

	if (_source.size() == 0)
		return;

	while ((int)start != (int)_source.npos)
	{

		start = stop + _delimiter.size();
		stop = _source.find(_delimiter, start);
		if ((int)stop == (int)_source.npos)
			break;

		length = stop - start;

		empty = ( (stop - start) == 1 );
		if (empty && _ignoreEmptyTokens)
			continue;

		_tokens.push_back(_source.substr(start, length));


		
	}

	_tokens.push_back(_source.substr(start, _source.npos));
}





vector<string> StringTokenizer::getAllTokens() {
	return _tokens;
}


int StringTokenizer::getNumberOfTokens() {
	return _tokens.size();
}


string StringTokenizer::getTokenAt(unsigned int index) {
	if (index < _tokens.size())
		return _tokens[index];

	throw MyException("ERROR: index out of bounds!");
}




void StringTokenizer::clear() {
	_tokens.clear();
	_source = "";
}


void StringTokenizer::setSource(string source) {
	_source = source;
}

void StringTokenizer::setDelimiter(string delimiter) {
	_delimiter = delimiter;
}


void StringTokenizer::setEmptyTokens(bool ignoreEmptyTokens) {
	_ignoreEmptyTokens = ignoreEmptyTokens;
}



void StringTokenizer::printAllTokens() {
	cout << setfill('0');
	for (unsigned int i = 0; i < _tokens.size(); i++)
		cout << "[" << setw(3) << i << "] = " << _tokens[i] << endl;
	cout << setfill(' ');
}





vector<string> StringTokenizer::justTokenize(string source, string delimiter, bool ignoreEmptyTokens) {
	StringTokenizer st(source, delimiter, ignoreEmptyTokens);
	st.tokenize();
	return st.getAllTokens();
}


