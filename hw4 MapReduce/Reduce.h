#include <vector>
#include <string>
#include <algorithm>

using namespace std;

typedef pair<string, int> Item;

bool ascending = true;

static bool comp(Item a, Item b) {
    if (ascending == true)
        return (a.first < b.first);
    else 
        return (a.first > b.first);
}

vector<Item> Sort(vector<Item> data)
{
    sort(data.begin(), data.end(), comp);
    return data;
}

struct classcomp {
    bool operator() (const string& lhs, const string& rhs) const {
        if (ascending == true)
            return lhs < rhs;
        else 
            return lhs > rhs;
    }
};

map<string, vector<int>, classcomp> Group(vector<Item> data)
{
    map<string, vector<int>, classcomp> grouped_data;
    for (auto it: data) {
        grouped_data[it.first].push_back(it.second);
    }
    return grouped_data;
}

Item Reduce(string key, vector<int> data)
{
    int sum = 0;
    for (auto it: data) {
        sum += it;
    }
    return make_pair(key, sum);
}

void Output(vector<Item> reduce_result, string output_dir, string job_name, int id)
{
    ofstream out(output_dir + job_name + "-" + to_string(id) + ".out");
    for (auto it: reduce_result) {
        out << it.first << " " << it.second << endl;
    }
}
