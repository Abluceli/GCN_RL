from __future__ import division
from __future__ import print_function


import tensorflow as tf
from gcn.models import *
from rl.models import *
from rl.replay_buffer import *
from rl.utils import *


def model_train(dataset):
    # Load data
    adj_norm_1, adj_norm_2, adj_1, adj_2, features, labels = load_data_rgcn(dataset)
    num_nodes = adj_norm_1[2][0]
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Define placeholders
    placeholders = get_placeholder()

    # Create Model
    model_rgcn_main = RGCN(placeholders, num_features, features_nonzero, "main")
    model_rgcn_target = RGCN(placeholders, num_features, features_nonzero, "target")
    state_representations_main = model_rgcn_main.outputs
    state_representations_target = model_rgcn_target.outputs

    model_rl_target = DQN(state=state_representations_target,
                          output_dim=num_nodes,
                          scope='target')
    model_rl_main = DQN(state=state_representations_main,
                        output_dim=num_nodes,
                        scope='main')


    # Create Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    target_update_op = get_target_update_op()

    # Training
    summary_writer = tf.summary.FileWriter(FLAGS.summary_path, graph=tf.get_default_graph())
    model_saver = tf.train.Saver(max_to_keep=1)
    replay_buffer = ReplayBuffer(buffer_size=FLAGS.buffer_size)
    frame_count = 0
    gcn_params = {'adj_norm_1': adj_norm_1,
                  'adj_norm_2': adj_norm_2,
                  'adj_1': adj_1,
                  'adj_2': adj_2,
                  'features': features,
                  'placeholders': placeholders,
                  'labels': labels}
    for epoch in range(FLAGS.epochs):
        # Update RL model
        episode_reward, episode_loss = run_training_episode(sess=sess,
                                                            model_rl_target=model_rl_target,
                                                            model_rl_main=model_rl_main,
                                                            gcn_params=gcn_params,
                                                            replay_buffer=replay_buffer,
                                                            target_update_op=target_update_op,
                                                            frame_count=frame_count,)
        print("Epoch: {}, Reward: {}, Loss: {}".format(epoch, episode_reward, episode_loss))
        summary = tf.Summary()
        summary.value.add(tag='rewards', simple_value=episode_reward)
        summary.value.add(tag='loss', simple_value=episode_loss)
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()
        if epoch % 50 == 0:
            model_saver.save(sess, FLAGS.model_path + '/model' + str(epoch) + '.cptk')


if __name__ == '__main__':
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')

    flags.DEFINE_integer('buffer_size', 10000, 'The maximum size of the replay buffer for DQN.')
    flags.DEFINE_integer('rl_episode_max_steps', 100, 'The maximum steps for rl agent ')
    flags.DEFINE_multi_float('epsilon', [1, 5000, 0.1],
                             ['Initial exploration rate', 'anneal steps', 'final exploration rate'])
    flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
    flags.DEFINE_integer('replay_start_size', 1, 'Number of experiences to be stored before training.')
    flags.DEFINE_integer('target_update_freq', 100, 'rl target network update frequency.')
    flags.DEFINE_integer('main_update_freq', 4, 'rl main network update frequency.')
    flags.DEFINE_integer('rl_batch_size', 2, 'Batch size for training rl.')
    flags.DEFINE_float('rl_lr', 0.0001, 'Initial rl learning rate.')

    flags.DEFINE_string('summary_path', './log', 'Path to store training summary.')
    flags.DEFINE_string('model_path', './model', 'Path to store model.')

    dataset = "BlogCatalog"
    model_train(dataset)


