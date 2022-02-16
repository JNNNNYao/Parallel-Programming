#include <map>
#include <queue>
#include <string>

using namespace std;

typedef pair<int, string> Record;

map<int, string> Input_split(int chunkID, string input_filename, int chunk_size)
{
    map<int, string> records;
    ifstream input_file(input_filename);
    int cnt = 0;
    string line;
    while (cnt != (chunkID-1)*chunk_size && getline(input_file, line)) {
        cnt++;  // move to desired position
    }
    while (cnt != (chunkID)*chunk_size && getline(input_file, line)) {
        records[cnt] = line;
        cnt++;
    }
    return records;
}

map<string, int> Map(Record record)
{
    size_t pos;
    string word;
    map<string, int> map_result;
    while ((pos = record.second.find(" ")) != string::npos) {
        word = record.second.substr(0, pos);

        if (map_result.count(word) == 0) {
            map_result[word] = 1;
        }
        else {
            map_result[word]++;
        }

        record.second.erase(0, pos + 1);
    }
    if (map_result.count(record.second) == 0) {
        map_result[record.second] = 1;
    }
    else {
        map_result[record.second]++;
    }
    return map_result;
}

int Partition(string key, int num_reducer) {
    int offset = key[0] - 'A';
    return offset % num_reducer;
}