import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from envir import Game

a = 1
x = Game()
max_steps = 4000
max_history = 20000
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
episode_reward = 0
episode_reward_history = []
point_history = []      # Histories hold information from previous frames, decisions
action_history = []
state_history = []
state_next_history = []
done_history = []
reward_history = []
update_after_actions = 5  # Specifies how often to update our model.
update_target_model = 10000


def create_model():
    inputs = layers.Input(shape=(3,))
    layer1 = layers.Dense(128, activation="relu")(inputs)
    action = layers.Dense(2, activation="softmax")(layer1)

    return keras.Model(inputs=inputs, outputs=action)


model = create_model()  # This is the main, decision making model
model.summary()
model_target = create_model()  # Target model, helps with learning
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_function = keras.losses.Huber()
episode_number = 0

while True:
    x.reset()
    state = x.state()
    frame = 0
    episode_reward = 0
    for step in range(1, max_steps):
        frame += 1
        if np.random.rand() < epsilon:
            action = np.random.choice(2)
        else:
            stateTensor = tf.convert_to_tensor(state)
            stateTensor = tf.expand_dims(stateTensor, 0)
            action_probs = model(stateTensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        x.main_loop()
        x.action(action)
        done = x.is_done()
        reward = x.get_reward()
        state_next = x.state()
        episode_reward += reward
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        reward_history.append(reward)
        done_history.append(done)
        state = state_next

        if frame % update_after_actions == 0 and len(done_history) > batch_size:
            # Chooses random samples from our history:
            indices = np.random.choice(range(len(done_history)), size=batch_size)
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            action_sample = np.array([action_history[i] for i in indices])
            reward_sample = np.array([reward_history[i] for i in indices])
            done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])
            future_rewards = model_target.predict(state_next_sample)
            updated_q_values = reward_sample + gamma * tf.reduce_max(future_rewards, axis=1)
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample
            masks = tf.one_hot(action_sample, 2)

            with tf.GradientTape() as tape:
                q_values = model(state_sample)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Updating target model with main model weights
        if frame % update_target_model == 0:
            model_target.set_weights(model.get_weights)

        # Deletes a row if we step over history limit
        if len(reward_history) > max_history:
            del reward_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if x.is_done():
            break
    episode_number += 1
    episode_reward_history.append(episode_reward)
    point_history.append(x.point)
    if len(episode_reward_history) > 10:
        del episode_reward_history[:1]
        del point_history[:1]
    mean_reward = np.mean(episode_reward_history)
    mean_point = np.mean(point_history)

    if episode_number % 5 == 0:
        print(episode_number, mean_reward, mean_point)
