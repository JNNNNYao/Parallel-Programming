#include <iostream>
#include <fstream>
#include <string>
#include <map>

using namespace std;

int main(int argc, char **argv)
{
    string job_name = string(argv[1]);
    int num_reducer = stoi(argv[2]);
    string ans_filename = string(argv[3]);

    map<string, int> result;
    for (int i = 0; i < num_reducer; i++) {
        string file = job_name + "-" + to_string(i+1) + ".out";
        ifstream input_file(file);
        string line;
        while (getline(input_file, line)) {
            size_t pos = line.find(" ");
            string key = line.substr(0, pos);
            int cnt = stoi(line.substr(pos+1));
            if (result.count(key) == 0) {
                result[key] = cnt;
            }
            else {
                result[key] += cnt;
            }
        }
    }
    string file = ans_filename;
    ifstream input_file(file);
    string line;
    while (getline(input_file, line)) {
        size_t pos = line.find(" ");
        string key = line.substr(0, pos);
        int cnt = stoi(line.substr(pos+1));
        if (result.count(key) == 0 || result[key] != cnt) {
            cout << "wrong answer!!!! count not match!!!!" << endl;
            cout << key << endl;
            return 0;
        }
        result.erase(key);
    }
    if (result.size() == 0) {
        cout << "pass!!!!" << endl;
    }
    else {
        cout << "wrong answer!!!! key not match!!!!" << endl;
    }
    return 0;
}