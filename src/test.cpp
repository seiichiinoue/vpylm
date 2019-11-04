#include "model.cpp"
using namespace std;

void test_train() {
    string dirname = "out";
	PyVPYLM* model = new PyVPYLM();
	model->load_textfile("data/processed/kokoro.txt", 0.95);
	model->set_g0(1.0 / model->get_num_types_of_words());

	for(int epoch=1; epoch<1000; ++epoch) {
		model->perform_gibbs_sampling();
		// model->sample_hyperparameters();
		double ppl = model->compute_perplexity_test();
		double log_likelihood = model->compute_log_Pdataset_test();
		cout << ppl << ", " << log_likelihood << endl;
	}
	model->save(dirname);
	model->load(dirname);
	model->remove_all_data();
	delete model;
}
void generate() {
	PyVPYLM *model = new PyVPYLM();
	model->load("./model/vpylm.model");
	wstring s = model->generate_sentence();
	wcout << s << endl;
}

int main(int argc, char *argv[]) {
	for(int i=0; i<1; ++i) {
		test_train();
	}
	generate();
}