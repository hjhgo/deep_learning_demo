import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
import tensorflow.contrib.tpu as tpu

def model_fn(features, labels, mode, params):
    def get_input(features, labels):
        input_x = tf.expand_dims(features, axis=-1)
        input_y = tf.squeeze(labels)

        return input_x, input_y

    def conv_net(x):
        initializer = tf.random_normal_initializer(stddev=0.01)
        with tf.name_scope("conv1"):
            filter = tf.get_variable("filter-%s" % 64, [5, 5, 1, 64],
                                     initializer=initializer)

            x = tf.nn.conv2d(x, filter=filter, strides=[1, 1, 1, 1], padding="SAME")
            x = tf.nn.elu(x)
            x = tf.nn.max_pool(x, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')
            if mode==tf.estimator.ModeKeys.TRAIN:
                with tf.name_scope("drop_out"):
                    x = tf.nn.dropout(x, keep_prob=params.drop_out)

        with tf.name_scope("conv2"):
            # conv
            filter = tf.get_variable("filter-%s" % 128, [5, 5, 64, 128],
                                     initializer=initializer)
            x = tf.nn.conv2d(x, filter=filter, strides=[1, 1, 1, 1], padding="SAME")
            x = tf.nn.elu(x)
            x = tf.nn.max_pool(x, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')
            if mode==tf.estimator.ModeKeys.TRAIN:
                with tf.name_scope("drop_out"):
                    x = tf.nn.dropout(x, keep_prob=params.drop_out)

#         with tf.name_scope("conv3"):
#             # conv
#             filter = tf.get_variable("filter-%s" % 258, [5, 5, 128, 258],
#                                      initializer=initializer)
#             x = tf.nn.conv2d(x, filter=filter, strides=[1, 1, 1, 1], padding="SAME")
#             x = tf.nn.elu(x)
#             x = tf.nn.max_pool(x, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')
#             if mode==tf.estimator.ModeKeys.TRAIN:
#                 with tf.name_scope("drop_out"):
#                     x = tf.nn.dropout(x, keep_prob=params.drop_out)

        return x

    def fullconnected(x):

        with tf.name_scope("full_connect1"):
            x = tf.layers.dense(x, units=params.hidden_size, activation="elu")
            if mode==tf.estimator.ModeKeys.TRAIN:
                with tf.name_scope("drop_out"):
                    x = tf.nn.dropout(x, keep_prob=params.drop_out)

        with tf.name_scope("full_connect2"):
            logist = tf.layers.dense(x, units=params.num_class)

        return logist

    x, y = get_input(features, labels)
    x = tf.contrib.layers.batch_norm(x)
    x = conv_net(x)
    x = layers.flatten(x)

    logits = fullconnected(x)
    prediction = tf.argmax(logits, axis=1)
    cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits))
    train_op = layers.optimize_loss(loss=cross_entropy,
                                    global_step=tf.train.get_global_step(),
                                    learning_rate=params.learning_rate,
                                    optimizer="Adam",
                                    clip_gradients=params.clip_max,
                                    )
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions={"logits": logits, "prediction": prediction},
                                      loss=cross_entropy,
                                      train_op=train_op,
                                      eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, prediction)})


# import tensorflow.contrib.training as train
# train.HParams

def create_estimator_and_spec():
    model_param = tf.contrib.training.HParams(
        num_class=10,
        hidden_size=256,
        clip_max=5.0,
        drop_out=0.7,
        learning_rate=0.01,
        batch_size=128
    )
    run_config = tf.estimator.RunConfig(
        model_dir="./",
        save_checkpoints_secs=300,
        save_summary_steps=100)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_param
    )

    return estimator


if __name__=='__main__':

    estimator = create_estimator_and_spec()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=x_train.astype(np.float32), y=y_train.astype(np.int32),
                                                        shuffle=True, batch_size=128, num_epochs=10)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x=x_test.astype(np.float32), y=y_test.astype(np.int32),
                                                       shuffle=False, batch_size=128, num_epochs=1)

    # estimator.train(input_fn=train_input_fn)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    evel_spec = tf.estimator.EvalSpec(input_fn=test_input_fn)
    #    estimator.train(input_fn=train_input_fn)
    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=evel_spec)
