import argparse, sys, os
sys.path.append(os.getcwd())
import model

def generate(args):
    vpylm = model.vpylm()
    vpylm.load(args.model)
    for _ in range(args.num):
        # print(hpylm.generate_sentence())
        # for japanese
        generated = vpylm.generate_sentence().replace(" ", "")
        print(generated)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=1)
    parser.add_argument("-m", "--model", default="./model")
    generate(parser.parse_args())