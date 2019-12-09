from __future__ import division
from __future__ import print_function

from gcn.utils import *
import numpy as np
import sys
sys.path.append('..')

flags = tf.app.flags
FLAGS = flags.FLAGS


def get_reward_simple(selected_list, labels):
    count = 0
    for id in selected_list:
        if labels[id] == 1:
            count += 1

    return count / len(selected_list)


def update_main_rl(sess,
                   model_rl_target,
                   model_rl_main,
                   replay_buffer):
    #[0:s, 1:a, 2:r, 3:s', 4:done]
    with sess. as_default():
        train_batch = replay_buffer.sample(size=FLAGS.rl_batch_size)
        # train_batch = np.moveaxis(train_batch, 0, 1)
        next_q_prime = sess.run(model_rl_target.qvalues,
                                feed_dict=train_batch[0, 3])
        next_q = sess.run(model_rl_main.qvalues,
                          feed_dict=train_batch[0, 3])
        target_qvalues = train_batch[0, 2] + (1-train_batch[0, 4])*FLAGS.gamma*next_q_prime[0, np.argmax(next_q[0])]
        print(target_qvalues)
        feed_dict = train_batch[0, 0]
        feed_dict.update({model_rl_main.chosen_actions: [train_batch[0, 1]],
                          model_rl_main.target_qvalues: [target_qvalues],
                          model_rl_main.lr: FLAGS.rl_lr})
        # feed_dict[model_rl_main.chosen_actions] = np.vstack(train_batch[0, 1])
        # feed_dict[model_rl_main.target_qvalues] = target_qvalues
        # feed_dict[model_rl_main.lr] = FLAGS.rl_lr
        loss, _ = sess.run([model_rl_main.loss,
                            model_rl_main.update],
                           feed_dict=feed_dict)
        return loss


def get_target_update_op():
    target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target')
    main_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main')
    op_holder = []
    for tp, mp in zip(target_params, main_params):
        op_holder.append(tp.assign(mp.value()))
    return op_holder


def run_training_episode(sess,
                         model_rl_target,
                         model_rl_main,
                         gcn_params,
                         replay_buffer,
                         target_update_op,
                         frame_count,
                         ):
    episode_reward = 0
    episode_losses = []
    selected_list = []
    with sess.as_default():
        feed_dict = construct_feed_dict(gcn_params['adj_norm_1'],
                                        gcn_params['adj_norm_2'],
                                        gcn_params['features'],
                                        FLAGS.dropout,
                                        gcn_params['placeholders'])
        # state = sess.run(model_gcn.outputs,
        #                  feed_dict=feed_dict)

        steps = 0
        while steps < FLAGS.rl_episode_max_steps:
            qvalues = sess.run(model_rl_main.qvalues, feed_dict=feed_dict)[0]

            (ep_start, anneal_steps, ep_end) = FLAGS.epsilon
            ratio = max((anneal_steps - max(frame_count-FLAGS.replay_start_size, 0))/float(anneal_steps), 0)
            ep = (ep_start - ep_end)*ratio + ep_end

            selected_node_id = np.random.choice(len(qvalues)) if np.random.rand() < ep else np.argmax(qvalues)
            r = get_reward_simple(selected_list, gcn_params['labels']) if steps == FLAGS.rl_episode_max_steps - 1 else 0
            if selected_node_id in selected_list:
                r -= 0.01

            selected_list.append(selected_node_id)
            gcn_params['adj_norm_1'], gcn_params['adj_norm_2'], gcn_params['adj_1'], gcn_params['adj_2'] \
                = update_adj(selected_node_id, gcn_params['adj_1'], gcn_params['adj_2'])


            episode_reward += r

            # check if done and get new state
            done = True if steps == FLAGS.rl_episode_max_steps-1 else False

            new_feed_dict = construct_feed_dict(gcn_params['adj_norm_1'],
                                                gcn_params['adj_norm_2'],
                                                gcn_params['features'],
                                                FLAGS.dropout,
                                                gcn_params['placeholders'])

            replay_buffer.add(np.reshape(np.array([feed_dict, selected_node_id, r, new_feed_dict, done]), [1, -1]))

            if frame_count > FLAGS.replay_start_size:
                if frame_count % FLAGS.main_update_freq == 0:
                    loss = update_main_rl(sess=sess,
                                          model_rl_target=model_rl_target,
                                          model_rl_main=model_rl_main,
                                          replay_buffer=replay_buffer)
                    episode_losses.append(loss)
                if frame_count % FLAGS.target_update_freq == 0:
                    sess.run(target_update_op)

            feed_dict = new_feed_dict
            frame_count += 1
            steps += 1

            print(steps)
            if done:
                break

        episode_loss = np.mean(episode_losses) if len(episode_losses) != 0 else 0
        return episode_reward, episode_loss
