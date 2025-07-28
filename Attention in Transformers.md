#Attention in Transformers

Before the Transformer architecture, many advanced models like **RNN**, **LSTM**, and **GRNN** were widely used for sequence tasks such as language translation and text generation. These models process data sequentially, step-by-step, which slows down training, especially for long sequences.

The **Transformer** model replaces this sequential processing with an **attention mechanism**, allowing for faster and more efficient training.

---

## Main Components of Transformers

### 1. Word Embedding

Word Embedding converts words (tokens) into vectors — series of numbers representing their meanings and relationships.

For example, the words **"boy"** and **"girl"** are mapped to vectors that reflect their semantic similarity. Vectors that are close in space imply related or similar meanings.

---

### 2. Positional Encoding

Since Transformers do not process words sequentially, they need a way to understand word order.

Consider the sentence:  
> *The cat sat on the mat*

Without positional encoding, the model sees the words but not their order. Positional encoding adds unique signals to each word's embedding so the model knows the sequence order — that **"cat"** comes before **"sat"**, etc.

---

### 3. Attention

Attention allows the model to focus on relevant words when processing a sentence.

In the sentence:  
> *The cat chased the mouse because it was hungry.*

The pronoun **"it"** could refer to **"cat"** or **"mouse"**. Attention helps the model decide that **"it"** refers to **"cat"** by analyzing context.

---

## What is Self-Attention?

Self-Attention compares each word in the sentence with every other word (including itself) and calculates similarity scores. These scores determine how much attention the model should pay to other words when encoding a given word.

### The Attention Formula

\[
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
\]

- **Q**: Query matrix  
- **K**: Key matrix  
- **V**: Value matrix  
- \(d_k\): dimension of the key vectors (used to scale the dot products)

---

## Example: "Write a poem"

1. Convert each word ("Write", "a", "poem") into embeddings.  
2. Add positional encodings to these embeddings.  
3. Multiply the encoded embeddings by learnable weight matrices to get **Q**, **K**, and **V** matrices.  
4. Calculate the attention scores using the formula above.  
5. The output is a weighted sum of the values (**V**), focusing on important parts of the sentence.

---

## Optional: Simple Python Illustration (Using NumPy)

```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Example Q, K, V matrices (3 words, embedding dim 4)
Q = np.array([[1,0,1,0], [0,1,0,1], [1,1,0,0]])
K = np.array([[1,0,1,0], [0,1,0,1], [1,1,0,0]])
V = np.array([[1,0,0,1], [0,2,1,0], [1,0,2,1]])

dk = Q.shape[-1]
scores = np.dot(Q, K.T) / np.sqrt(dk)
weights = softmax(scores)
output = np.dot(weights, V)

print("Attention Weights:\n", weights)
print("Output:\n", output)
