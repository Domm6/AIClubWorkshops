import tensorflow as tf
import numpy as np
import os

tf.test.gpu_device_name()

text = open("/text.txt", "r").read()
print(text[:200])

vocab = sorted(set(open(text, "r").read()))
print("Number of Unique Characters: {}".format(len(vocab)))

char2Index = {c: i for i, c in enumerate(vocab)}
int_text = np.array([char2Index[i] for i in text])

index2Char = np.array(vocab)

print("Character to Index: \n")
for i in char2Index:
    print("  {:4s}: {:3d}".format(repr(i), char2Index[i]))

print("\nInput text to Integer: \n")
print('{} mapped to {}'.format(repr(text[:20]), int_text[:20]))

seq_length = 150  # max number of characters that can be fed per input

char_dataset = tf.data.Dataset.from_tensor_slices(int_text)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


def create_input_target_pair(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(create_input_target_pair)

BATCH_SIZE = 64

# Buffer used for shuffling dataset
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

print(dataset)

vocab_size = len(vocab)
# Embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))

dir_checkpoints = './training_checkpoints'
checkpoint_prefix = os.path.join(dir_checkpoints, "checkpoint_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True)

EPOCHS = 10

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(dir_checkpoints)


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(dir_checkpoints))
model.build(tf.TensorShape([1, None]))

model.summary()


def generate_text(model, start_string):
    num_generate = 1000  # Number of characters to be generated

    input_eval = [char2Index[s]
                  for s in start_string]  # Converting input to indexes
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.5

    # Reset any hidden states in the model
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Use a categorical distribution to get the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # We set the predicted character as the next input to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(index2Char[predicted_id])

    return (start_string + ''.join(text_generated))


test = input("Enter your starting string: ")
print(generate_text(model, start_string=test))
