import argparse, sys, os, codecs, random
import model

def train(args):
    try:
        os.mkdir(args.model)
    except:
        pass
    vpylm = model.vpylm()
    vpylm.set_seed(0)
    vpylm.load_textfile(args.filename, args.split_ratio)
    # logging
    print("train data size: {}".format(vpylm.get_num_train_data()))
    print("test data size: {}".format(vpylm.get_num_test_data()))
    print("vocablary: {}".format(vpylm.get_num_types_of_words()))
    print("num of total words: {}".format(vpylm.get_num_words()))

    # set base distribution
    vpylm.set_g0(1.0/float(vpylm.get_num_types_of_words()))
    vpylm.prepare()

    # training
    for epoch in range(1, args.epoch+1):
        vpylm.perform_gibbs_sampling()
        vpylm.sample_hyperparams()
        if epoch % 100 == 0:
            # validation
            print("epoch: {}/{}".format(epoch, args.epoch))
            print("train: likelihood: {} perplexity: {}".format(vpylm.compute_log_Pdataset_train(), vpylm.compute_perplexity_train()))
            print("test: likelihood: {} perplexity: {}".format(vpylm.compute_log_Pdataset_test(), vpylm.compute_perplexity_test()))
            vpylm.save(args.model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", default="./data/processed/kokoro.txt")
    parser.add_argument("-e", "--epoch", type=int, default=10000)
    parser.add_argument("-m", "--model", default="./model")
    parser.add_argument("-r", "--split_ratio", type=float, default=0.8)
    train(parser.parse_args())