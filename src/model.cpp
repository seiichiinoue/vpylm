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
    // statistics
    unordered_map<id, int> _word_count;
    int _sum_word_count;
    bool gibbs_first_addition;
    PyVPYLM() {
        setlocale(LC_CTYPE, "ja_JP.UTF-8");
        ios_base::sync_with_stdio(false);
        locale default_loc("ja_JP.UTF-8");
        locale::global(default_loc);
        locale ctype_default(locale::classic(), default_loc, locale::ctype);
        wcout.imbue(ctype_default);
        wcin.imbue(ctype_default);

        _vpylm = new VPYLM();
        _vocab = new Vocab();
        _gibbs_first_addition = true;
        _sum_word_count = 0;
    }
    ~PyVPYLM() {
        delete _vpylm;
        delete _vocab;
    }
    bool load_textfile(string filename, double split_ratio) {
        wifstream ifs(filename.c_str());
        wstring sentence;
        if (ifs.fail()) {
            return false;
        }
        vector<wstring> lines;
        while (getline(ifs, sentence) && !sentence.empty()) {
            lines.push_back(sentence);
        }
        vector<int> rand_indices;
        for (int i=0; i<lines.size(); ++i) {
            rand_indices.push_back(i);
        }
        int split = lines.size() * split_ratio;
        shuffle(rand_indices.begin(), rand_indices.end(), sampler::mt);
        for (int i=0; i<rand_indices.size(); ++i) {
            wstring &sentence = lines[rand_indices[i]];
            if (i < split) {
                add_train_data(sentence);
            } else {
                add_test_data(sentence);
            }
        }
        return true;
    }
    void add_train_data(wstring sentence) {
        _add_data_to(sentence, _dataset_train);
    }
    void add_test_data(wstring sentence) {
        _add_data_to(sentence, _dataset_test);
    }
    void _add_data_to(wstring &sentence, vector<vector<id>> &dataset) {
        vector<wstring> word_str_array;
        split_word_by(sentence, L' ', word_str_array);
        if (word_str_array.size() > 0) {
            vector<id> words;
            for (int i=0; i<_vpylm->_depth; ++i) {
                words.push_back(ID_BOS);
            }
            for (auto word_str : word_str_array) {
                if (word_str.size() == 0) {
                    continue;
                }
                id token_id = _vocab->add_string(word_str);
                words.push_back(token_id);
                _word_count[token_id] += 1;
                _sum_word_count += 1;
            }
            words.push_back(ID_EOS);
            dataset.push_back(words);
        }
    }
    void prepare() {
        for (int data_index=0; data_index<_dataset_train.size(); ++data_index) {
            vector<id> &token_ids = _dataset_train[data_index];
            vector<int> prev_depths(token_ids.size(), -1);
            _prev_depths_for_data.push_back(prev_depths);
        }
    }
    void set_g0(double g0) {
        _vpylm->_g0 = g0;
    }
    void set_seed(int seed) {
        sampler::mt.seed(seed);
    }
    void load(string dir) {
        _vocab->load(dir+"/vpylm.vocab");
        if (_vpylm->load(dir+"/vpylm.model")) {
            _gibbs_first_addition = false;
        }
    }
    void save(string dir) {
        _vocab->save(dir+"/vpylm.vocab");
        _vpylm->save(dir+"/vpylm.model");
    }
};