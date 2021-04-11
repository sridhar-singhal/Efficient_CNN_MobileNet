#include <bits/stdc++.h>
using namespace std;

signed main()
{
    string inp;

    ifstream ReadFile("proto.txt");

    while (getline (ReadFile, inp)) {
        cout<<inp<<"\n";
    }

    istringstream ss(inp);
  
    string word;
    vector<string> vs;

    while (ss >> word) 
    {
        vs.push_back(word);
    }

    string kernel_name=vs[0];
    string output_size=vs[1];
    string filt_size=vs[2];
    string pad=vs[3];
    string stride=vs[4];

    ReadFile.close();
}