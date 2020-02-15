import fasttext
import re
import os

class word2vec:
    def train_model(self):
        model_train = fasttext.train_unsupervised('./data/text_comments.txt',
                                                  "skipgram",
                                                  thread=4,
                                                  epoch=10,
                                                  lr=.02)
        model_train.save_model("./data/skip.bin")
        return model_train


    def generate_output(self, model):
        (output_dim, feature_dim) = model.get_output_matrix().shape
        word_list = model.get_words()

        with open('./data/word_vecs.txt', 'w', encoding='utf-8') as fp:
            print("writing to output...")
            fp.write("%d %d\n" % (output_dim, feature_dim))
            for word in word_list:
                s = str(word).encode('utf8')
                fp.write('%s ' % s.decode('utf8'))

                s_list = str(list(model.get_word_vector(word))).strip("[]")
                r_sub = re.sub(",", "", s_list)
                fp.write(r_sub)
                fp.write('\n')
            print("writing done.")
            fp.close()

if __name__ == '__main__':
    w2v = word2vec()
    model = w2v.train_model()
    # model = fasttext.load_model("./output/cbow.bin")

    # generate_output(model)
