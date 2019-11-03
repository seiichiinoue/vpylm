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
    double g0;                  // 0-gram probabiliry
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
};