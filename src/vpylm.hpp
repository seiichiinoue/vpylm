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
};