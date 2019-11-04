#include <boost/python.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <string>
#include <unordered_map> 
#include "node.hpp"
#include "vpylm.hpp"
#include "vocab.hpp"
using namespace boost;

void split_word_by(const wstring &str, wchar_t delim, vector<wstring> &elems) {
    elems.clear();
    wstring item;
    for (wchar_t ch : str) {
        if (ch == delim) {
            if (!item.empty()) {
                elems.push_back(item);
            }
            item.clear();
        } else {
            item += ch;
        }
    }
    if (!item.empty()) {
        elems.push_back(item);
    }
}

template<class T>
python::list list_from_vector(vector<T> &vec){  
     python::list list;
     typename vector<T>::const_iterator it;

     for(it = vec.begin(); it != vec.end(); ++it)   {
          list.append(*it);
     }
     return list;
}

class PyVPYLM {
public:
    VPYLM *_vpylm;
    Vocab *_vocab;
    vector<vector<id>> _dataset_train;
    vector<vector<id>> _dataset_test;
    vector<vector<int>> _prev_depths_for_data;
    vector<int> _rand_indices;
};