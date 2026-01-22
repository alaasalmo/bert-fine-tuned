<p><cneter><img src=img\fine-tuning.jpg></cneter></p>

# Fine tuning with Bert Transformer

Fine-tuning is the process of training a Large Language Model (LLM) such as LLAMA, BERT, or ChatGPT on a specialized domain or task. These models begin as pre trained models, meaning they have already learned general language patterns from large datasets. After pre training, we can fine tune a model on a smaller, domain specific dataset to adapt it for a particular application. For example, we can take a base model like BERT and fine-tune it for a targeted task or specialized area.

In this article, we will focus on the BERT model, a widely used pre trained Transformer model. We will examine its architecture in detail and compare BERT with other models such as LLAMA and ChatGPT.

BERT’s key architectural feature is its bidirectionality, which comes from the self-attention layers within its Transformer blocks. Unlike traditional models such as GPT, where attention is unidirectional (tokens attend only left to right), BERT’s self attention allows each token to attend to all other tokens in the input sequence both left and right. BERT is primarily a Seq2Vector model, whereas ChatGPT is a Seq2Seq model. This article will also explore Seq2Vector behavior in depth.

BERT was introduced by the Google research team in 2018. Its development follows a two-stage process: (1) pre-training on large unlabeled datasets, and (2) fine tuning on smaller, task-specific datasets. In this article, we will focus on the pre-trained BERT model and then apply fine-tuning techniques to structured data for three tasks: data completeness checks, data classification, and data clustering.

While these tasks can be performed using traditional machine-learning methods, BERT provides significant advantages when structured datasets contain text fields, sentences, or multiple languages. In such cases, BERT’s language-understanding capabilities offer important benefits in model performance and flexibility.

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model developed by Google in 2018.

## I.  The Architecture diagram of BERT model 

<!--<img src="img/arch-fune-tuning-3.jpg"></div> -->
Before diving into fine tuning, we have to understand the archtecture diagram for Bert.

<div align="center"><img src="img/bert-arch.png"></div>

According to the diagram above, BERT has different layers like encoding, attention and classification layers

#### 1. Input Layer

The input is tokenized text, broken into tokens (words, sub words, or characters)

#### 2. Encoding Layer

This layer transforms the input tokens into dense vector representations (embeddings). After the encoding, we should specify the position for each token. After that we have to combine the the embeding with position

Please check the diagram below:

<div align="center"><img src="img/steps-encodinging.png"></div>


#### 3. Transformer blocks (architecture)
A. Attention and multi-attention layer

- Attention Layer (Self-Attention)
Each word attends to (looks at) all other words in the sentence to understand context. 

Why we call it Self-Attention: Queries, Keys, and Values all come from the same sentence and each word attends to other words in itself

In Scaled Dot-Product Attention (the core inside MHA), the attention weights are computed. By using three values Q, K and V
See this example. In real life the three values with have matrix to calculate the weight. 

Q (Query): “What am I looking for?”

K (Key): “What features do you have?”

V (Value): “What content should I propagate based on attention?”

Intuitively: the query asks a question, the keys provide context, and the values provide the actual information to mix.

In the example below explain more:

<div align="center"><img src="img/self-attention.png"></div>

- Multi-attention
In Bert each self-Attension is header. Bert can run many headers in parallel.   

#### B. Residual + LayerNorm

<u>Residual</u> (add) connections:

The core problem: why gradients struggle in deep networks. In very deep models (like Transformers with 12–96 layers):
Gradients are computed using backpropagation. Each layer multiplies gradients by weights and derivatives

To solve this issue, we will do the following steps:

1- Group the layers to Block. The block contains many layers

2- We create to paths First path is regulare path way or longer pathway Second one is short cut pathway (skip connections)

3- The input in each block is the addtion of short cut pathway and long pathway (Addition math or concat). Math is better because in concat, the size of layer will change. In the transfoermer, we use math addition

Advantage:

1- Better Gradian flow

2- Faster Learner 

3- Enables deeper learners


<p><cneter><img src=img\Residual-Connections.jpg></cneter></p>

<u>LayerNorm</u> fixes this immediately, ensuring:

- stable training
- faster convergence

Why we need Normolization during training:
After residual,the Values can get too large or too small. This causes unstable gradients and slow learning. For this reaso, the normolization fixes the issue. 

The steps of normolization is:

Step 1: Find the mean

Step 2: Subtract the mean (center the values)

Step 3: Find the variance

Step 4: Find the standard deviation

Step 5: Divide by the standard deviation (normalize)

#### C. Feed-Forward Network (FFN)

Before explaining FFNs, let’s quickly review the structure of a neural network. A neural network is typically made up of three parts: an input layer, one or more hidden layers, and an output layer. Each node (neuron) in the hidden layer applies either a linear or non-linear function to the input it receives.
An FFN is the simplest type of artificial neural network. In an FFN, data flows in only one direction: from the input → through the hidden layers → to the output. 

There are no loops or feedback connections.
Each node in a layer performs a linear transformation (matrix multiplication plus bias), followed by a non-linear activation function such as ReLU, sigmoid, or tanh.
The information always moves forward only; it never cycles back. Because of this, FFNs have no memory of past inputs—they process each input independently.
Another important property of FFNs is that they allow parallel processing, making them very well-suited for GPU acceleration.

In the diagram below shows one example of FFN with input, hidden layers and output. Also it shows different activation functions for each neuron.
<p><cneter><img src=img\ffn.jpg></cneter></p>

After we explain about the architecture of transformer layer blocks. We will have two kinds of blocks in the Bert transformer.
BERT has 12 transformer blocks in BERT-Base, and 24 in BERT-Large.

#### 5. Output Layer

Final prediction based on the task:
Class probabilities (e.g., positive vs. negative sentiment),
Answer spans (start/end positions in QA),
Next sentence prediction (original pretraining objective).

## III.The Seq2Vector in BERT

Seq2Vector is a process for BERT transformer. Seq2Vector (sequence-to-vector): Input is a sequence → output is a single vector (fixed-size embedding).
This is useful for classification, clustering, or similarity tasks, where you don’t need another sequence as output. Seq2Vector is not good in translation process like English to French for example. Seq2Vector is different than Seq2Seq. ChatGPT and LLAMA are Seq2Seq. this mean these two models are good for translation.  


Example: ["the", "animal", "it"]

1- In Bert: [CLS] the animal it [SEP]
2- So after self-attention, you get:
	3 tokens → 3 vectors
	"the"    → v₁
	"animal" → v₂
	"it"     → v₃
3- Attention with weights:
the    → 0.1
animal → 0.7
it     → 0.2

4- Vector output:

output = 0.1·v_the + 0.7·v_animal + 0.2·v_it

## IV. The BERT types
### A. BERT Base & Large

<b>BERT-Base</b><br>
12 Transformer encoder layers<br>
768 hidden size (embedding dimension)<br>
12 attention heads<br>
~110M parameters<br>
<b>BERT-Large</b><br>
24 Transformer encoder layers<br>
1024 hidden size<br>	
16 attention heads<br>
~340M parameters<br>

Because of BERT Base is lighter than BERT Large, for this reason the BERT Large is more accurate in the result than BERT Base.

### B. DistilBERT

DistilBERT is a smaller, faster, and lighter version of BERT designed to retain most of BERT’s language understanding capabilities while being more efficient. It was introduced by Hugging Face in 2019.
Now we can go deeper to understand the differences between Bert-base and DistilBERT. 

Bert has 12 encoder layers (blocks) while Distlbert has 6 encoder layer (blocks). 

The DistilBERT has three main parts: 

A. Teacher Bert. It’s base Bert. 

B. Student Bert. It’s Distal Bert. 

C. Three Loss functions (Distillation Loss, Consine Loss and MLM Loss) 


<b>Distillation loss</b> is the loss used in knowledge distillation, where a large model (teacher) trains a smaller model (student) by transferring its "knowledge.". Instead of training only on the correct labels, the student learns from the teacher's output probabilities (soft predictions).

<b>Consine loss</b> measures how similar two vectors are in direction. 

<b>MLM Loss</b> teaches the model to: Understand context, Learn grammar + meaning and Predict words from surrounding text

<p><cneter><img src=img\distil-bert-teacher-student.jpg></cneter></p>

## V.  Examples of BERT in python

<p>