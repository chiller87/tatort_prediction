

#ifndef __TEST_TOKENIZER_H__
#define __TEST_TOKENIZER_H__


#include "StringTokenizer.h"
#include <iostream>


using namespace std;

void test_empty_string();
void test_without_delimiter();
void test_with_delimiter();
void test_empty_token();
void test_token_at_begin();
void test_token_at_end();
void test_abstract_tokenize();





void test_empty_string() {
	cout << "TESTING: test_empty_string" << endl;
	string str = "";
	StringTokenizer st(str, "<A>", false);
	st.tokenize();
	int count = st.getNumberOfTokens();

	cout << "String = '" << str << "', delimiter = '<A>'" << endl;
	cout << "expected size of tokens '0' and got '" << count << "'" << endl;
	if (count == 0)
		cout << "====> test_empty_string     PASS" << endl << endl;
	else
		cout << "====> test_empty_string     FAIL" << endl << endl;



}


void test_without_delimiter() {
	cout << "TESTING: test_without_delimiter" << endl;
	string str = "foo bar baz";
	StringTokenizer st(str, "<A>", false);
	st.tokenize();
	int count = st.getNumberOfTokens();

	cout << "String = '" << str << "', delimiter = '<A>'" << endl;
	cout << "expected size of tokens '1' and got '" << count << "'" << endl;
	st.printAllTokens();
	if (count == 1)
		cout << "====> test_without_delimiter     PASS" << endl << endl;
	else
		cout << "====> test_without_delimiter     FAIL" << endl << endl;
}


void test_with_delimiter() {
	cout << "TESTING: test_with_delimiter" << endl;
	string str = "foo<A>bar<A>baz";
	StringTokenizer st(str, "<A>", false);
	st.tokenize();
	int count = st.getNumberOfTokens();

	cout << "String = '" << str << "', delimiter = '<A>'" << endl;
	cout << "expected size of tokens '3' and got '" << count << "'" << endl;
	st.printAllTokens();
	if (count == 3)
		cout << "====> test_with_delimiter     PASS" << endl << endl;
	else
		cout << "====> test_with_delimiter     FAIL" << endl << endl;
}


void test_empty_token() {
	cout << "TESTING: test_empty_token" << endl;
	string str = "foo<A><A>bar<A>baz";
	StringTokenizer st(str, "<A>", false);
	st.tokenize();
	int count = st.getNumberOfTokens();

	cout << "String = '" << str << "', delimiter = '<A>'" << endl;
	cout << "expected size of tokens '3' and got '" << count << "'" << endl;
	st.printAllTokens();
	if ((count == 4) && (st.getTokenAt(1) == ""))
		cout << "====> test_empty_token     PASS" << endl << endl;
	else
		cout << "====> test_empty_token     FAIL" << endl << endl;
}




void test_token_at_begin() {
	cout << "TESTING: test_token_at_begin" << endl;
	string str = "<A>foo<A>bar<A>baz";
	StringTokenizer st(str, "<A>", false);
	st.tokenize();
	int count = st.getNumberOfTokens();

	cout << "String = '" << str << "', delimiter = '<A>'" << endl;
	cout << "expected size of tokens '4' and got '" << count << "'" << endl;
	st.printAllTokens();
	if ((count == 4) && (st.getTokenAt(0) == ""))
		cout << "====> test_token_at_begin     PASS" << endl << endl;
	else
		cout << "====> test_token_at_begin     FAIL" << endl << endl;
}

void test_token_at_end() {
	cout << "TESTING: test_token_at_end" << endl;
	string str = "foo<A>bar<A>baz<A>";
	StringTokenizer st(str, "<A>", false);
	st.tokenize();
	int count = st.getNumberOfTokens();

	cout << "String = '" << str << "', delimiter = '<A>'" << endl;
	cout << "expected size of tokens '4' and got '" << count << "'" << endl;
	st.printAllTokens();
	if ((count == 4) && (st.getTokenAt(3) == ""))
		cout << "====> test_token_at_end     PASS" << endl << endl;
	else
		cout << "====> test_token_at_end     FAIL" << endl << endl;
}


void test_abstract_tokenize() {
	cout << "TESTING: test_abstract_tokenize" << endl;
	string str = "foo<A>bar<A>baz<A>";

	vector<string> res = StringTokenizer::justTokenize(str, "<A>", false);
	int count = res.size();

	cout << "String = '" << str << "', delimiter = '<A>'" << endl;
	cout << "expected size of tokens '4' and got '" << count << "'" << endl;
	if ((count == 4) && (res[3] == ""))
		cout << "====> test_abstract_tokenize     PASS" << endl << endl;
	else
		cout << "====> test_abstract_tokenize     FAIL" << endl << endl;
}


#endif


