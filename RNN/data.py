# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# ---- Phase 1: Dataset Creation ----
# Educational Text Classification Dataset
texts = [
    "The Pythagorean theorem is used in right-angled triangles.",  # Math
    "Einstein’s theory of relativity revolutionized our understanding of space and time.",  # Science
    "The Battle of Hastings took place in 1066 between the Normans and Saxons.",  # History
    "Calculus is a branch of mathematics that studies continuous change.",  # Math
    "Newton discovered the laws of motion and gravity.",  # Science
    "The industrial revolution began in the 18th century in Britain.",  # History
]

labels = ['Math', 'Science', 'History', 'Math', 'Science', 'History']

# Map labels to integers
label_map = {'Math': 0, 'Science': 1, 'History': 2}
labels = [label_map[label] for label in labels]

# ---- Phase 2: Text Preprocessing ----
# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences to ensure uniform input size
X_class = pad_sequences(sequences, padding='post')

# Convert labels to one-hot encoding
y_class = to_categorical(labels, num_classes=len(label_map))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.33, random_state=42)

# ---- Phase 3: Model Building for Classification ----
def build_classification_model(input_shape):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=input_shape[0]))
    model.add(SimpleRNN(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes (Math, Science, History)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the classification model
classification_model = build_classification_model(X_train.shape[1:])
classification_model.summary()

classification_model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# ---- Phase 4: Evaluate the Classification Model ----
classification_accuracy = classification_model.evaluate(X_test, y_test)
print(f'Classification Model Test Accuracy: {classification_accuracy[1]:.2f}')

# ---- Phase 5: Next Word Generation Dataset ----
# Create a larger corpus for next word generation (Math-related content for demonstration)
corpus = [
    "Mathematics is the study of numbers, shapes, and patterns.",
    "Calculus deals with rates of change and slopes of curves.",
    "Algebra involves variables and equations.",
    "Geometry involves the study of shapes, sizes, and positions.",
    "Mathematics is essential in fields like physics and engineering."
]

# Tokenizenizer_gen = Tokenizer()
tokenizer_gen.fit_on_texts(corpus)

sequences = tokenizer_gen.texts_to_sequences(corpus)

# Prepare the sequences: each sequence is a set of words, and the next word is the target
input_sequences = []
next_words = []
for seq in sequences:
    for i in range(1, len(seq)):
        input_sequences.append(seq[:i])
        next_words.append(seq[i])

# Pad sequences
input_sequences = pad_sequences(input_sequences, padding='pre')

# Convert next words to one-hot encoding
next_words = to_categorical(next_words, num_classes=len(tokenizer_gen.word_index) + 1)

# ---- Phase 6: Model Building for Next Word Generation ----
def build_generation_model(input_shape):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer_gen.word_index) + 1, output_dim=64, input_length=input_shape[0]))
    model.add(SimpleRNN(64, activation='relu'))
    model.add(Dense(len(tokenizer_gen.word_index) + 1, activation='softmax'))  # Output layer: softmax for next word prediction
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the next word generation model
generation_model = build_generation_model(input_sequences.shape[1:])
generation_model.summary()

generation_model.fit(input_sequences, next_words, epochs=10, batch_size=2)

# ---- Phase 7: Demonstrating Next Word Generation ----
def generate_text(model, tokenizer, seed_text, num_words):
    # Generate the next words based on the input seed text
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([seed_text])
        sequence = pad_sequences(sequence, maxlen=model.input_shape[1], padding='pre')
        predicted_word_idx = np.argmax(model.predict(sequence), axis=-1)
        predicted_word = tokenizer.index_word[predicted_word_idx[0]]
        seed_text += ' ' + predicted_word
    return seed_text

# Test the generation model with a starting sentence
start_sentence = "Mathematics is the study of"
generated_text = generate_text(generation_model, tokenizer_gen, start_sentence, num_words=20)
print(f"Generated text: {generated_text}")

