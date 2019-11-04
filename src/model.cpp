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
    bool _gibbs_first_addition;
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
    void perform_gibbs_sampling() {
        if (_rand_indices.size() != _dataset_train.size()) {
            _rand_indices.clear();
            for (int data_index=0; data_index<_dataset_train.size(); ++data_index) {
                _rand_indices.push_back(data_index);
            }
        }
        shuffle(_rand_indices.begin(), _rand_indices.end(), sampler::mt);
        for (int n=0; n<_dataset_train.size(); ++n) {
            int data_index = _rand_indices[n];
            vector<id> &token_ids = _dataset_train[data_index];
            vector<int> &prev_depths = _prev_depths_for_data[data_index];
            for (int token_t_index=1; token_t_index<token_ids.size(); ++token_t_index) {
                if (_gibbs_first_addition == false) {
                    int prev_depth = prev_depths[token_t_index];
                    _vpylm->remove_customer_at_timestep(token_ids, token_t_index, prev_depth);
                }
                int new_depth = _vpylm->sample_depth_at_timestep(token_ids, token_t_index);
                _vpylm->add_customer_at_timestep(token_ids, token_t_index, new_depth);
                prev_depths[token_t_index] = new_depth;
            }
        }
        _gibbs_first_addition = false;
    }
    void remove_all_data() {
        for (int i=0; i<_dataset_train.size(); ++i) {
            vector<id> &token_ids = _dataset_train[i];
            vector<int> &prev_depths = _prev_depths_for_data[i];
            // token_ids = [BOS, ids_1, ids_2, ..., ids_t]
            for (int j=1; j<token_ids.size(); ++j) {
                int prev_depth = prev_depths[j];
                _vpylm->remove_customer_at_timestep(token_ids, j, prev_depth);
            }
        }
    }
    int get_num_train_data() {
        return _dataset_train.size();
    }
    int get_num_test_data() {
        return _dataset_test.size();
    }
    int get_num_nodes() {
        return _vpylm->get_num_nodes();
    }
    int get_num_customers() {
        return _vpylm->get_num_customers();
    }
    int get_num_types_of_words() {
        return _word_count.size();
    }
    int get_num_words() {
        return _sum_word_count;
    }
    int get_vpylm_depth() {
        return _vpylm->get_depth();
    }
    id get_bos_id() {
        return ID_BOS;
    }
    id get_eos_id() {
        return ID_EOS;
    }
};