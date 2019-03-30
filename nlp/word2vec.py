import tensorflow as tf
import numpy as np
import jieba

# skim-gram
flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_integer(name="window_size", default=4, help="word to vec skip window size")
flags.DEFINE_integer(name="vocabulary_size", default=50, help="word to vec skip window size")
flags.DEFINE_integer(name="embedding_size", default=20, help="word to vec skip window size")


class Word2Vec():
    def __init__(self):
        self.window_size = FLAGS.window_size
        self.vocabulary_size = FLAGS.vocabulary_size
        self.embedding_size = FLAGS.embedding_size
        self.build_model()

    def build_model(self):
        self.input_x = tf.placeholder(shape=[None], dtype=tf.int32)
        self.input_y = tf.placeholder(shape=[None,self.window_size], dtype=tf.int32)
        #

        with tf.name_scope(name="input_wights"):
            w_in = tf.get_variable(name="word_input_vec", shape=[self.vocabulary_size, self.embedding_size],
                                   initializer=tf.random_normal_initializer(stddev=0.1))

        # tf.gather
        self.input_x_emb = tf.nn.embedding_lookup(params=w_in, ids=self.input_x)

        with tf.name_scope(name="output_wights"):
            w_out = tf.get_variable(name="vec_out_word", shape=[self.vocabulary_size, self.embedding_size])
            w_out_biases = tf.get_variable(name="vec_out_word_bais", shape=[self.vocabulary_size])

        self.loss = tf.nn.nce_loss(weights=w_out,
                              biases=w_out_biases,
                              inputs=self.input_x_emb,
                              labels=self.input_y,
                              num_sampled=1,
                              num_classes=self.vocabulary_size,
                              num_true=4,
                              )
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        self.top_3 = tf.nn.top_k(tf.squeeze(tf.matmul(w_in,self.input_x_emb,transpose_b=True)),k=3)


    def train(self):
        x,y = self.get_data()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        epoch = 20
        for i in range(epoch):
            _,l = sess.run([self.train_op,self.loss],feed_dict={self.input_x:x,self.input_y:y})
        res = sess.run(self.top_3,feed_dict={self.input_x:[9]})
        print(res)
        print(self.words[9],"最近的词:")
        for i in res.indices:
            print(self.words[i])
        # 最近的词
        print(res)
        sess.close()


    def get_data(self):
        x = "我们都有一个家，他的名字叫中国,都有一个家，都有都有"
        self.word_dirt ={}
        x = list(jieba.cut(x))
        self.word_dirt["unk"] = 0
        self.words = []
        self.words.append("unk")
        index=  1
        for word in x:
            if word not in self.word_dirt.keys():
                self.word_dirt[word] = index
                index += 1
                self.words.append(word)
                if index>=FLAGS.vocabulary_size:
                    break
        print(self.word_dirt)
        input_x = []
        for word  in x:
            input_x.append(self.word_dirt.get(word,0))
        # print(input_x)
        spilt_x = []
        # widow_size = 4
        spilt_y = []
        size = len(input_x)
        for i in range(2,size-2):
            spilt_x.append(input_x[i])
            temp = []
            for j in range(i-2,i):
                temp.append(input_x[j])
            for j in range(i+1,i+3):
                temp.append(input_x[j])
            spilt_y.append(temp)

        return spilt_x,spilt_y


w = Word2Vec()
w.train()
