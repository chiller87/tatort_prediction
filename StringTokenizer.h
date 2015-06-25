#ifndef __STRINGTOKENIZER_H__
#define __STRINGTOKENIZER_H__

#include <string>
#include <vector>

using namespace std;



class StringTokenizer {
private:
	
	string _source;
	string _delimiter;

	vector<string> _tokens;

	int _count;

	bool _ignoreEmptyTokens;

public:
	StringTokenizer(string source = "", string delimiter = ",", bool ignoreEmptyTokens = false);


	static vector<string> justTokenize(string source, string delimiter = ",", bool ignoreEmptyTokens = false);


	


	void tokenize();
	vector<string> getAllTokens();
	int getNumberOfTokens();
	string getTokenAt(unsigned int index);
	void clear();
	void setSource(string source);
	void setDelimiter(string delimiter);
	void setEmptyTokens(bool ignoreEmptyTokens);

	void printAllTokens();

};









#endif
