#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <unordered_map> 
#include <unordered_set>
#include <vector>
#include <cassert>
#include <fstream>
#include "sampler.hpp"
#include "common.hpp"
#include "node.hpp"

class VPYLM {
public:
    Node *_root;
    int _depth;
    double _g0;                  // 0-gram probabiliry
    // params of m-depth node
    vector<double> _d_m;
    vector<double> _theta_m;
    // for estimating parameters of distribution
    vector<double> _a_m;
    vector<double> _b_m;
    vector<double> _alpha_m;
    vector<double> _beta_m;
    double _beta_stop;
    double _beta_pass;
    // for speeding up calculation
    int _max_depth;
    double *_sampling_table;

    VPYLM() {
        _root = new Node(0);
        _root->_depth = 0;
        _beta_stop = VPYLM_BETA_STOP;
        _beta_pass = VPYLM_BETA_PASS;
        _max_depth = 999;
        _sampling_table = new double[_max_depth];
    }
    ~VPYLM() {
        _delete_node(_root);
        delete[] _sampling_table;
    }
    void _delete_node(Node *node) {
        for (auto &elem : node->_children) {
            Node *child = elem.second;
            _delete_node(child);
        }
        delete node;
    }
    bool add_customer_at_timestep(vector<id> &token_ids, int token_t_index, int depth_t) {
        Node *node = find_node_by_tracing_back_context(token_ids, token_t_index, depth_t, true);
        id token_t = token_ids[token_t_index];
        return node->add_customer(token_t, _g0, _d_m, _theta_m);
    }
    bool remove_customer_at_timestep(vector<id> &token_ids, int token_t_index, int depth_t) {
        Node *node = find_node_by_tracing_back_context(token_ids, token_t_index, depth_t, true);
        id token_t = token_ids[token_t_index];
        node->remove_customer(token_t);
        if (node->need_to_remove_from_parent()) {
            node->remove_from_parent();
        }
        return true;
    }
    // tracing back from `t` by `order_t`
    // token_ids:           [0, 1, 2, 3, 4, 5]
    // token_t_index: 4            ^     ^
    // order_t: 2                  |<- <-|
    Node *find_node_by_tracing_back_context(vector<id> &token_ids, int token_t_index, int order_t, bool generate_node_if_needed=false, bool return_middle_node=false) {
        if (token_t_index - order_t < 0) {
            return NULL;
        }
        Node *node = _root;
        for (int depth=1; depth<=order_t; ++depth) {
            id context_token_id = token_ids[token_t_index - depth];
            Node *child = node->find_child_node(context_token_id, generate_node_if_needed);
            if (child == NULL) {
                if (return_middle_node) {
                    return node;
                }
                return NULL;
            }
            node = child;
        }
        return node;
    }
    int sample_depth_at_timestep(vector<id> &context_token_ids, int token_t_index) {
        if (token_t_index == 0) {
            return 0;
        }
        id token_t = context_token_ids[token_t_index];
        // censoring if stop prob below this value
        double eps = 1e-24;
        double sum = 0;
        double p_pass = 1;
        double pw = 0;
        int sampling_table_size = 0;
        Node *node = _root;
        for (int n=0; n<=token_t_index; ++n) {
            if (node) {
                pw = node->compute_Pw(token_t, _g0, _d_m, _theta_m);
                double p_stop = node->stop_probability(_beta_stop, _beta_pass, false) * p_pass;
                double p = pw * p_stop;
                p_pass *= node->pass_probability(_beta_stop, _beta_pass, false);
                _sampling_table[n] = p;
                sampling_table_size += 1;
                sum += p;
                if (p_stop < eps) {
                    break;
                }
                if (n < token_t_index) {
                    id context_token_id = context_token_ids[token_t_index - n - 1];
                    node = node->find_child_node(context_token_id);
                }
            } else {
                double p_stop = p_pass * _beta_stop / (_beta_stop + _beta_pass);
                double p = pw * p_stop;
                _sampling_table[n] = p;
                sampling_table_size += 1;
                sum += p;
                p_pass *= _beta_pass / (_beta_stop + _beta_pass);
                if (p_stop < eps) {
                    break;
                }
            }
        }
        double normalizer = 1.0 / sum;
        double bernoulli = sampler::uniform(0, 1);
        double stack = 0;
        for (int n=0; n<sampling_table_size; ++n) {
            stack += _sampling_table[n] * normalizer;
            if (bernoulli < stack) {
                return n;
            }
        }
        return _sampling_table[sampling_table_size - 1];
    }
    double compute_Pw_given_h(id token_id, vector<id> &context_token_ids) {
        Node *node = _root;
        // censoring if stop prob below this value
        double eps = 1e-24;
        double parent_pw = _g0;
        double p_pass = 1;
        double p_stop = 1;
        double pw_h = 0;
        // position of token_id is depth 0
        int depth = 0;
        while (p_stop > eps) {
            // if context node does not exit
            if (node == NULL) {
                p_stop = p_pass * _beta_stop / (_beta_stop + _beta_pass);
                pw_h += parent_pw * p_stop;
                p_pass *= _beta_pass / (_beta_stop + _beta_pass);
            } else {
                double pw = node->compute_Pw_with_parent_Pw(token_id, parent_pw, _d_m, _theta_m);
                p_stop = node->stop_probability(_beta_stop, _beta_pass, false) * p_pass;
                p_pass *= node->pass_probability(_beta_stop, _beta_pass, false);
                pw_h += pw * p_stop;
                parent_pw = pw;
                if (depth < context_token_ids.size()) {
                    id context_token_id = context_token_ids[context_token_ids.size() - depth - 1];
                    node = node->find_child_node(context_token_id);
                } else {
                    node = NULL;
                }
            }
            depth++;
        }
        return pw_h;
    }
    double compute_Pn_given_h(int n, vector<id> &context_token_ids) {
        Node *node = _root;
        double p_stop, p_pass;
        for (int depth=0; depth<=n; ++depth) {
            if (node == NULL) {
                p_stop = p_pass * _beta_stop / (_beta_stop + _beta_pass);
                p_pass *= _beta_pass / (_beta_stop + _beta_pass);
            } else {
                p_stop = node->stop_probability(_beta_stop, _beta_pass);
                p_pass = node->pass_probability(_beta_stop, _beta_pass);
                if (depth < context_token_ids.size()) {
                    id context_token_id = context_token_ids[context_token_ids.size() - depth - 1];
                    node = node->find_child_node(context_token_id);
                } else {
                    node = NULL;
                }
            }
        }
        return p_stop;
    }
    double compute_Pw(vector<id> &token_ids) {
        if (token_ids.size() == 0) {
            return 0;
        }
        double mult_pw = 1;
        vector<id> context_token_ids(token_ids.begin(), token_ids.begin() + 1);
        for (int t=1; t<token_ids.size(); ++t) {
            id token_id = token_ids[t];
            double pw_h = compute_Pw_given_h(token_id, context_token_ids);
            mult_pw *= pw_h;
            context_token_ids.push_back(token_ids[t]);
        }
        return mult_pw;
    }
    double compute_log_Pw(vector<id> &token_ids) {
        if (token_ids.size() == 0) {
            return 0;
        }
        double sum_pw_h = 0;
        vector<id> context_token_ids(token_ids.begin(), token_ids.begin() + 1);
        for (int t=1; t<token_ids.size(); ++t) {
            id token_id = token_ids[t];
            double pw_h = compute_Pw_given_h(token_id, context_token_ids);
            sum_pw_h += log(pw_h);
            context_token_ids.push_back(token_ids[t]);
        }
        return sum_pw_h;
    }
    double compute_log2_Pw(vector<id> &token_ids) {
        if (token_ids.size() == 0) {
            return 0;
        }
        double sum_pw_h = 0;
        vector<id> context_token_ids(token_ids.begin(), token_ids.begin() + 1);
        for (int t=1; t<token_ids.size(); ++t) {
            id token_id = token_ids[t];
            double pw_h = compute_Pw_given_h(token_id, context_token_ids);
            sum_pw_h += log2(pw_h);
            context_token_ids.push_back(token_ids[t]);
        }
        return sum_pw_h;
    }
    void init_hyperparams_at_depth_if_needed(int depth) {
        if (depth >= _d_m.size()) {
            while (_d_m.size() <= depth) {
                _d_m.push_back(HPYLM_INITIAL_D);
            }
        }
        if (depth >= _theta_m.size()) {
            while (_theta_m.size() <= depth) {
                _theta_m.push_back(HPYLM_INITIAL_THETA);
            }
        }
        if (depth >= _a_m.size()) {
            while (_a_m.size() <= depth) {
                _a_m.push_back(HPYLM_INITIAL_A);
            }
        }
        if (depth >= _b_m.size()) {
            while (_b_m.size() <= depth) {
                _b_m.push_back(HPYLM_INITIAL_B);
            }
        }
        if (depth >= _alpha_m.size()) {
            while (_alpha_m.size() <= depth) {
                _alpha_m.push_back(HPYLM_INITIAL_ALPHA);
            }
        }
        if (depth >= _beta_m.size()) {
            while (_beta_m.size() <= depth) {
                _beta_m.push_back(HPYLM_INITIAL_BETA);
            }
        }
    }
    void sum_auxiliary_variables_recursively(Node *node, vector<double> &sum_log_x_u_m, vector<double> &sum_y_ui_m, vector<double> &sum_1_y_ui_m, vector<double> &sum_1_z_uwkj_m) {
        for (auto elem : node->_children) {
            Node *child = elem.second;
            int depth = child->_depth;
            double d = _d_m[depth];
            double theta = _theta_m[depth];
            sum_log_x_u_m[depth] += child->auxiliary_log_x_u(theta);    // log(x_u)
            sum_y_ui_m[depth] += child->auxiliary_y_ui(d, theta);       // y_ui
            sum_1_y_ui_m[depth] += child->auxiliary_1_y_ui(d, theta);   // 1 - y_ui
            sum_1_z_uwkj_m[depth] += child->auxiliary_1_z_uwkj(d);      // 1 - z_uwkj
            sum_auxiliary_variables_recursively(child, sum_log_x_u_m, sum_y_ui_m, sum_1_y_ui_m, sum_1_z_uwkj_m);
        }
    }
    // estimating `d` and `theta`
    void sample_hyperparams() {
        int max_depth = get_depth();
        vector<double> sum_log_x_u_m(max_depth + 1, 0.0);
        vector<double> sum_y_ui_m(max_depth + 1, 0.0);
        vector<double> sum_1_y_ui_m(max_depth + 1, 0.0);
        vector<double> sum_1_z_uwkj_m(max_depth + 1, 0.0);
        // root
        sum_log_x_u_m[0] = _root->auxiliary_log_x_u(_theta_m[0]);
        sum_y_ui_m[0] = _root->auxiliary_y_ui(_d_m[0], _theta_m[0]);
        sum_1_y_ui_m[0] = _root->auxiliary_1_y_ui(_d_m[0], _theta_m[0]);
        sum_1_z_uwkj_m[0] = _root->auxiliary_1_z_uwkj(_d_m[0]);
        // others
        init_hyperparams_at_depth_if_needed(max_depth);
        sum_auxiliary_variables_recursively(_root, sum_log_x_u_m, sum_y_ui_m, sum_1_y_ui_m, sum_1_z_uwkj_m);
        for (int u=0; u<=max_depth; ++u) {
            _d_m[u] = sampler::beta(_a_m[u] + sum_1_y_ui_m[u], _b_m[u] + sum_1_z_uwkj_m[u]);
            _theta_m[u] = sampler::gamma(_alpha_m[u] + sum_y_ui_m[u], _beta_m[u] - sum_log_x_u_m[u]);

        }
    }
    int get_num_nodes() {
        return _root->get_num_nodes();
    }
    int get_num_customers() {
        return _root->get_num_customers();
    }
    int get_num_tables() {
        return _root->get_num_tables();
    }
    int get_sum_stop_counts() {
        return _root->sum_stop_counts();
    }
    int get_sum_pass_counts() {
        return _root->sum_pass_counts();
    }
    void update_max_depth(Node *node, int &max_depth) {
        if (node->_depth > max_depth) {
            max_depth = node->_depth;
        }
        for (const auto &elem : node->_children) {
            Node *child = elem.second;
            update_max_depth(child, max_depth);
        }
    }
    int get_depth() {
        int max_depth = 0;
        update_max_depth(_root, max_depth);
        return max_depth;
    }
    void count_tokens_of_each_depth(unordered_map<int, int> &map) {
        _root->count_tokens_of_each_depth(map);
    }
    void enumerate_phrases_at_depth(int depth, vector<vector<id>> &phrases) {
        vector<Node*> nodes;
        _root->enumerate_nodes_at_depth(depth, nodes);
        for (auto &node : nodes) {
            vector<id> phrase;
            while (node->_parent) {
                phrase.push_back(node->_token_id);
                node = node->_parent;
            }
            phrases.push_back(phrase);
        }
    }
    template <class Archive>
    void serialize(Archive& archive, unsigned int version)
    {
        archive & _root;
        archive & _g0;
        archive & _beta_stop;
        archive & _beta_pass;
        archive & _d_m;
        archive & _theta_m;
        archive & _a_m;
        archive & _b_m;
        archive & _alpha_m;
        archive & _beta_m;
    }
    bool save(string filename = "hpylm.model"){
        std::ofstream ofs(filename);
        boost::archive::binary_oarchive oarchive(ofs);
        oarchive << *this;
        return true;
    }
    bool load(string filename = "hpylm.model"){
        std::ifstream ifs(filename);
        if(ifs.good() == false){
            return false;
        }
        boost::archive::binary_iarchive iarchive(ifs);
        iarchive >> *this;
        return true;
    }
};