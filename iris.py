import tensorflow as tf
from sklearn.datasets import load_iris
import tensorboard
import tensorflow.contrib.layers as layers
# tf.train.CheckpointSaverListener
d= load_iris()
x = d.data
y = d.target

class ExampleCheckpointSaverListener(tf.train.CheckpointSaverListener):

    def begin(self):
        print("start the session")

    def before_save(self, session, global_step_value):
        print(global_step_value)
        print('About to write a checkpoint')

    def after_save(self, session, global_step_value):
        print('after to write a checkpoint')
        # if global_step_value>100:
        return True

    def end(self, session, global_step_value):
        print("end the session")





# C:\Users\hj\AppData\Local\conda\conda\envs\py3.6\Lib\site-packages\tensorflow\__init__.py
def my_model(features,labels,mode,params):
    steps = tf.Variable(0,trainable=False)
    x = features["x"]
    y = labels
    for num_unit in params["num_units"]:
        x = tf.layers.dense(x,num_unit,activation="elu")
    logits = tf.layers.dense(x,params["num_classes"],activation=None)

    prediction = tf.cast(tf.argmax(tf.nn.softmax(logits=logits,axis=1),axis=1),tf.int32)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,y),tf.float32))
    accuracy = tf.metrics.accuracy(prediction,y)
    loss =tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y,logits=logits))
    train_op = layers.optimize_loss(loss=loss,optimizer=tf.train.AdamOptimizer,learning_rate=0.01,clip_gradients=5.0,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops={"accuracy":accuracy}
                                      )

def train():
    # 设置log级别
    tf.logging.set_verbosity(tf.logging.INFO)

    # 运行参数
    config = tf.estimator.RunConfig(model_dir="./output",
                                    # 每多少步去保存一个模型
                                    save_checkpoints_steps=300,
                                    # 每**步写入摘要
                                    save_summary_steps=100,
                                    # 打出log
                                    log_step_count_steps=50
                                    )
    # 模型参数
    params = {"batch_size":256,
              "num_units":[10,10],
              "num_classes":3}

    estimator = tf.estimator.Estimator(model_fn=my_model,
                           config= config,
                           params= params)

    train_input = tf.estimator.inputs.numpy_input_fn(x={"x":x},y = y,num_epochs=500,shuffle=True)
    # 在写入保存模型前后的操作。
    saver = ExampleCheckpointSaverListener()
    # tf.train.SessionRunHook()
    # ?
    estimator.train(train_input,saving_listeners=[saver])
    acc  = estimator.evaluate(train_input)
    print(acc)
    # eval_input = tf.estimator.inputs.numpy_input_fn(x={},y = )
if __name__ == '__main__':
    train()
