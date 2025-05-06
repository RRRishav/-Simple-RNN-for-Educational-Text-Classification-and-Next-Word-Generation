# -Simple-RNN-for-Educational-Text-Classification-and-Next-Word-Generation



## **Project Overview**

This project explores the power of **Recurrent Neural Networks (RNNs)** to perform two important NLP tasks using educational content:

### 1. **Educational Text Classification**  
The task is to classify short educational text snippets into categories such as **Math**, **Science**, **History**, and **English**.

### 2. **Next Word Generation**  
The model generates the next 20 words in a sentence, allowing for the simulation of intelligent **auto-completion** or **content prediction**. This is based on a larger educational corpus related to a specific field (Math, Science, or History).

---

## **Features**

- **Text Classification:** Classifies educational text into categories such as Math, Science, History, and English.
- **Next Word Generation:** Predicts the next 20 words based on a given sentence, demonstrating predictive text capabilities.

---

## **Dataset Creation**

### 1. **Text Classification Dataset:**
- A custom dataset consisting of educational text snippets in the fields of **Math**, **Science**, **History**, and **English**.
- Each snippet is labeled with one of the categories, and the model learns to classify the text based on these labels.

### 2. **Next Word Generation Dataset:**
- A large corpus of educational content related to **Math**, **Science**, or **History**.
- This dataset is used to train the model to predict the next word in a sequence, aiding the task of intelligent auto-completion.

---

## **Preprocessing Steps**

### **Text Classification Preprocessing:**
1. **Tokenization:** Text snippets are tokenized into words, and a vocabulary is created.
2. **Sequence Encoding:** Text is converted into integer sequences.
3. **Padding:** Sequences are padded to a fixed length to ensure consistent input size.
4. **Label Encoding:** The category labels are one-hot encoded.
5. **Train-Test Split:** The dataset is split into training and testing sets.

### **Next Word Generation Preprocessing:**
1. **Tokenization:** The educational text corpus is tokenized to create sequences of words.
2. **Sequence Creation:** Each sequence consists of a series of words with the next word as the target.
3. **Numerical Representation:** Words are converted into numerical tokens using the vocabulary.
4. **Target Encoding:** The next word in the sequence is one-hot encoded.

---

## **Model Architecture**

### **1. Classification Model:**
- **Embedding Layer:** Converts word indices into dense vectors.
- **RNN Layer:** A simple **RNN** layer processes the input sequence.
- **Dense Layer:** A fully connected layer with a softmax activation function for multi-class classification.

### **2. Generation Model:**
- **Embedding Layer:** Converts word indices into dense vectors.
- **RNN Layer:** A simple **RNN** layer processes sequences to predict the next word.
- **Dense Layer:** A fully connected layer with a softmax activation function to predict the next word in the sequence.

---

## **Training and Evaluation**

### **1. Training the Classification Model:**
- The model is trained using the educational text dataset with a set number of epochs.
- The classification model is evaluated on a test set to report the accuracy.

### **2. Training the Generation Model:**
- The model is trained on sequences of words, with each sequence used to predict the next word.
- The next word generation is evaluated by providing a starting sentence and generating the next 20 words.

---

## **Usage**

### **Text Classification:**
To classify a text snippet:

```python
# Example text snippet
text = "The solar system consists of the Sun and the objects that orbit it."

# Predicting the category
category = classify_text(text)
print(f"Predicted Category: {category}")
