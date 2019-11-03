#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <unordered_map>
#include <string>
#include <vector>
#include <cassert>
#include <fstream>
#include "common.hpp"
#include "sampler.hpp"

using namespace std;

class Node {
public:
    hashmap<id, Node*> _children;
    hashmap<id, vector<int>> _arrangement;
    Node *_parent;
    int _num_tables;
    int _num_customers;
    int _stop_count;
    int _pass_count;
    int _depth;
    id _token_id;
    Node(id token_id=0) {
        _num_tables = 0;
        _num_customers = 0;
        _stop_count = 0;
        _pass_count = 0;
        _token_id = token_id;
        _parent = NULL;
    }
    bool add_customer_to_table(id token_id, int table_k, double g0, vector<double> &d_m, vector<double> &theta_m) {
        auto itr = _arrangement.find(token_id);
        if (itr == _arrangement.end()) {
            return add_customer_to_new_table(token_id, g0, d_m, theta_m);
        } // else
        vector<int> &num_customers_at_table = itr->second;
        num_customers_at_table[table_k]++;
        _num_customers++;
        return true;
    }
    bool add_customer_to_new_table(id token_id, double g0, vector<double> &d_m, vector<double> &theta_m) {
        auto itr = _arrangement.find(token_id);
        if (itr == _arrangement.end()) {
            vector<int> tables = {1};
            _arrangement[token_id] = tables;
        } else {
            vector<int> &num_customers_at_table = itr->second;
            num_customers_at_table.push_back(1);
        }
        _num_tables++;
        _num_customers++;
        if (_parent != NULL) {
            // send dummy customer to parent node(restraunt)
            _parent->add_customer(token_id, g0, d_m, theta_m, false);
        }
        return true;
    }
    bool remove_customer_from_table(id token_id, int table_k) {
        auto itr = _arrangement.find(token_id);
        vector<int> &num_customers_at_table = itr->second;
        num_customers_at_table[table_k]--;
        _num_customers--;
        if (num_customers_at_table[table_k] == 0) {
            if (_parent != NULL) {
                _parent->remove_customer(token_id, false);
            }
            num_customers_at_table.erase(num_customers_at_table.begin() + table_k);
            _num_tables--;
            if (num_customers_at_table.size() == 0) {
                _arrangement.erase(token_id);
            }
        }
        return true;
    }
    bool add_customer(id token_id, double g0, vector<double> &d_m, vector<double> &theta_m, bool update_beta_count=true) {
        init_hyperparams_at_depth_if_needed(_depth, d_m, theta_m);
        double d_u = d_m[_depth];
        double theta_u = theta_m[_depth];
        double parent_Pw = g0;
        if (_parent) {
            parent_Pw = _parent->compute_Pw(token_id, g0, d_m, theta_m);
        }
        auto itr = _arrangement.find(token_id);
        /* add customer to new table */
        if (itr == _arrangement.end()) {
            add_customer_to_new_table(token_id, g0, d_m, theta_m);
            if (update_beta_count) {
                increment_stop_count();
            }
            return true;
        }
        /* add customer to existing table */
        // num of customer per table
        vector<int> &num_customers_at_table = itr->second;
        // for normalizing prob, calculate summation of all prob
        double sum = 0;
        for (int k=0; k<num_customers_at_table.size(); ++k) {
            sum += std::max(0.0, num_customers_at_table[k] - d_u);
        }
        double t_u = _num_tables;
        sum += (theta_u + d_u * t_u) * parent_Pw;
        double normalizer = 1.0 / sum;
        double bernoulli = sampler::uniform(0, 1);
        double stack = 0;
        // calculate probabiliry of adding customer to all of table serving `w`
        for (int k=0; k<num_customers_at_table.size(); ++k) {
            stack += std::max(0.0, num_customers_at_table[k] - d_u) * normalizer;
            if (bernoulli <= stack) {
                add_customer_to_table(token_id, k, g0, d_m, theta_m);
                if (update_beta_count) {
                    increment_stop_count();
                }
                return true;
            }
        }
        add_customer_to_new_table(token_id, g0, d_m, theta_m);
        if (update_beta_count) {
            // increment stop count
        }
        return true;
    }
    bool remove_customer(id token_id, bool update_beta_count=true) {
        auto itr = _arrangement.find(token_id);
        // num of customer per table
        vector<int> &num_customers_at_table = itr->second;
        // for normalizer 
        double sum = std::accumulate(num_customers_at_table.begin(), num_customers_at_table.end(), 0);
        double normalizer = 1.0 / sum;
        double bernoulli = sampler::uniform(0, 1);
        double stack = 0;
        // c_{u w k}; num of customer at table k of restaurant u serving word w
        for (int k=0; k<num_customers_at_table.size(); ++k) {
            stack += num_customers_at_table[k] * normalizer;
            if (bernoulli <= stack) {
                remove_customer_from_table(token_id, k);
                if (update_beta_count) {
                    decrement_stop_count();
                }
                return true;
            }
        }
        remove_customer_from_table(token_id, num_customers_at_table.size() - 1);
        if (update_beta_count) {
            decrement_stop_count();
        }
        return true;
    }
    double compute_Pw(id token_id, double g0, vector<double> &d_m, vector<double> &theta_m) {
        init_hyperparams_at_depth_if_needed(_depth, d_m, theta_m);
        double d_u = d_m[_depth];
        double theta_u = theta_m[_depth];
        double t_u = _num_tables;
        double c_u = _num_customers;
        auto itr = _arrangement.find(token_id);
        /* if target token does not exist */
        if (itr == _arrangement.end()) {
            double coeff = (theta_u + d_u * t_u) / (theta_u + c_u);
            if (_parent != NULL) {
                return _parent->compute_Pw(token_id, g0, d_m, theta_m) * coeff;
            }
            // if _parent == NULL then;
            return g0 * coeff;
        }
        /* target token exist*/
        double parent_Pw = g0;
        if (_parent != NULL) {
            // calculate recursively if parent does exist!
            parent_Pw = _parent->compute_Pw(token_id, g0, d_m, theta_m);
        }
        vector<int> &num_customers_at_table = itr->second;
        // c_uw: aggregate num of customer at all table of restaurant u serving word w
        double c_uw = std::accumulate(num_customers_at_table.begin(), num_customers_at_table.end(), 0);
        // t_uw: aggregate num of table at restaurant u serving word w
        double t_uw = num_customers_at_table.size();
        double first_term = std::max(0.0, c_uw - d_u * t_uw) / (theta_u + c_u);
        double second_coeff = (theta_u + d_u * t_u) / (theta_u + c_u);
        return first_term + second_coeff * parent_Pw;
    }
    // avoiding recursive calculation of parent Pw with serving parent Pw in advance
    double compute_Pw_with_parent_Pw(id token_id, double parent_pw, vector<double> &d_m, vector<double> &theta_m) {
        init_hyperparams_at_depth_if_needed(_depth, d_m, theta_m);
        double d_u = d_m[_depth];
        double theta_u = theta_m[_depth];
        double t_u = _num_tables;
        double c_u = _num_customers;
        auto itr = _arrangement.find(token_id);
        /* if target token does not exist */
        if (itr == _arrangement.end()) {
            double coeff = (theta_u + d_u * t_u) / (theta_u + c_u);
            return parent_pw * coeff;
        }
        vector<int> &num_customers_at_table = itr->second;
        // c_uw: aggregate num of customer at all table of restaurant u serving word w
        double c_uw = std::accumulate(num_customers_at_table.begin(), num_customers_at_table.end(), 0);
        // t_uw: aggregate num of table at restaurant u serving word w
        double t_uw = num_customers_at_table.size();
        double first_term = std::max(0.0, c_uw - d_u * t_uw) / (theta_u + c_u);
        double second_coeff = (theta_u + d_u * t_u) / (theta_u + c_u);
        return first_term + second_coeff * parent_pw;
    }
    // vpylm
    double stop_probability(double beta_stop, double beta_pass, bool recursive=true) {
        double p = (_stop_count + beta_stop) / (_stop_count + _pass_count + beta_stop + beta_pass);
        if (!recursive) {
            return p;
        }
        if (_parent != NULL) {
            p *= _parent->pass_probability(beta_stop, beta_pass);
        }
        return p;
    }
    double pass_probability(double beta_stop, double beta_pass, bool recursive=true) {
        double p = (_pass_count + beta_pass) / (_stop_count + _pass_count + beta_stop + beta_pass);
        if (!recursive) {
            return p;
        }
        if (_parent != NULL) {
            p *= _parent->pass_probability(beta_stop, beta_pass);
        }
        return p;
    }
    void increment_stop_count() {
        _stop_count++;
        if (_parent != NULL) {
            _parent->increment_pass_count();
        }
    }
    void decrement_stop_count() {
        _stop_count--;
        if (_parent != NULL) {
            _parent->decrement_pass_count();
        }
    }
    void increment_pass_count() {
        _pass_count++;
        if (_parent != NULL) {
            _parent->increment_pass_count();
        }
    }
    void decrement_pass_count() {
        _pass_count--;
        if (_parent != NULL) {
            _parent->decrement_pass_count();
        }
    }
    // for estimating `d` and `theta`; supplementary variable
    // x_u, y_ui, z_uwkj
<<<<<<< HEAD
    double auxiliary_log_x_u(double theta_u) {
        if(_num_customers >= 2) {
            double x_u = sampler::beta(theta_u + 1, _num_customers - 1);
            return log(x_u + 1e-8);
        }
        return 0;
    }
    double auxiliary_y_ui(double d_u, double theta_u) {
        if(_num_tables >= 2) {
            double sum_y_ui = 0;
            for(int i = 1;i <= _num_tables - 1;i++) {
                double denominator = theta_u + d_u * i;
                assert(denominator > 0);
                sum_y_ui += sampler::bernoulli(theta_u / denominator);
            }
            return sum_y_ui;
        }
        return 0;
    }
    double auxiliary_1_y_ui(double d_u, double theta_u) {
        if(_num_tables >= 2) {
            double sum_1_y_ui = 0;
            for(int i = 1;i <= _num_tables - 1;i++) {
                double denominator = theta_u + d_u * i;
                assert(denominator > 0);
                sum_1_y_ui += 1.0 - sampler::bernoulli(theta_u / denominator);
            }
            return sum_1_y_ui;
        }
        return 0;
    }
    double auxiliary_1_z_uwkj(double d_u) {
        double sum_z_uwkj = 0;
        // c_u..
        for(auto elem: _arrangement) {
            // c_uw.
            vector<int> &num_customers_at_table = elem.second;
            for(int k = 0;k < num_customers_at_table.size();k++) {
                // c_uwk
                int c_uwk = num_customers_at_table[k];
                if(c_uwk >= 2){
                    for(int j = 1;j <= c_uwk - 1;j++) {
                        assert(j - d_u > 0);
                        sum_z_uwkj += 1 - sampler::bernoulli((j - 1) / (j - d_u));
                    }
                }
            }
        }
        return sum_z_uwkj;
    }
=======
	double auxiliary_log_x_u(double theta_u) {
		if(_num_customers >= 2) {
			double x_u = sampler::beta(theta_u + 1, _num_customers - 1);
			return log(x_u + 1e-8);
		}
		return 0;
	}
	double auxiliary_y_ui(double d_u, double theta_u) {
		if(_num_tables >= 2) {
			double sum_y_ui = 0;
			for(int i = 1;i <= _num_tables - 1;i++) {
				double denominator = theta_u + d_u * i;
				assert(denominator > 0);
				sum_y_ui += sampler::bernoulli(theta_u / denominator);
			}
			return sum_y_ui;
		}
		return 0;
	}
	double auxiliary_1_y_ui(double d_u, double theta_u) {
		if(_num_tables >= 2) {
			double sum_1_y_ui = 0;
			for(int i = 1;i <= _num_tables - 1;i++) {
				double denominator = theta_u + d_u * i;
				assert(denominator > 0);
				sum_1_y_ui += 1.0 - sampler::bernoulli(theta_u / denominator);
			}
			return sum_1_y_ui;
		}
		return 0;
	}
	double auxiliary_1_z_uwkj(double d_u) {
		double sum_z_uwkj = 0;
		// c_u..
		for(auto elem: _arrangement) {
			// c_uw.
			vector<int> &num_customers_at_table = elem.second;
			for(int k = 0;k < num_customers_at_table.size();k++) {
				// c_uwk
				int c_uwk = num_customers_at_table[k];
				if(c_uwk >= 2){
					for(int j = 1;j <= c_uwk - 1;j++) {
						assert(j - d_u > 0);
						sum_z_uwkj += 1 - sampler::bernoulli((j - 1) / (j - d_u));
					}
				}
			}
		}
		return sum_z_uwkj;
	}
>>>>>>> b70357286981743bb175a097540df71ddcbc9262
    void init_hyperparams_at_depth_if_needed(int depth, vector<double> &d_m, vector<double> &theta_m) {
        if (depth >= d_m.size()) {
            while (d_m.size() <= depth) {
                d_m.push_back(HPYLM_INITIAL_D);
            }
            while (theta_m.size() <= depth) {
                theta_m.push_back(HPYLM_INITIAL_THETA);
            }
        }
<<<<<<< HEAD
    }
    template <class Archive>
    void serialize(Archive& archive, unsigned int version)
    {
        archive & _children;
        archive & _arrangement;
        archive & _num_tables;
        archive & _num_customers;
        archive & _parent;
        archive & _stop_count;
        archive & _pass_count;
        archive & _token_id;
        archive & _depth;
    }
    friend ostream& operator<<(ostream& os, const Node& node){
        os << "[id." << node._token_id << ":depth." << node._depth << "]" << endl;
        os << "_num_tables: " << node._num_tables << ", _num_customers: " << node._num_customers << endl;
        os << "_stop_count: " << node._stop_count << ", _pass_count: " << node._pass_count << endl;
        os << "- _arrangement" << endl;
        for(auto elem: node._arrangement){
            os << "  [" << elem.first << "]" << endl;
            os << "    ";
            for(auto customers: elem.second){
                os << customers << ",";
            }
            os << endl;
        }
        os << endl;
        os << "- _children" << endl;
        os << "    ";
        for(auto elem: node._children){
        }
        os << endl;
        return os;
=======
>>>>>>> b70357286981743bb175a097540df71ddcbc9262
    }
};