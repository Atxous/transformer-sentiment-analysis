# transformer-sentiment-analysis
Using a transformer's encoder, I attempt to train a model to undergo sentiment analysis based on IMDB's movie reviews dataset.

PyTorch project no. 4
# Background

I've seen how transformers work and how they act, but I was intrigued by how I would be able to use them for sentiment analysis. Though I could have used a pretrained model such as BERT, I thought I would do the favor to myself of creating my own. I ran into a lot of hurdles along the way, as I slowly realized how hard this task would be, but I ultimately succeeded in what I think is an okay model. There's lots of things I would like to improve on, but I believe at this time I'm satisfied with the results.

This is probably my most technically challenging project I've made so far. I had no idea how to get started when it came to preprocessing text in PyTorch and creating a transformer encoder. In Tensorflow, all I'd have to do is slap a TextVectorization layer before anything and that'd be it. However, I will probably keep using PyTorch because of the control it offers me, and I feel like I actually know what I'm doing while coding in it.

# How it works

PyTorch offers its built in **TransformerEncoder** class, but I wanted to build my own using the **MultiHeadAttention** layer. So this is pretty raw code with **PositionalEncoding** based on the sin and cosine waves as well as the rest of the encoder in the **TransformerEncoder** class.

I had to set the number of heads to 1 because training took much longer with 2 or 4 heads. I initially tried to use a gradient accumulation technique so that I could simulate higher batches, but this proved ineffective upon training time. For reference, I had to go down to 16 or 8 batches when I used 2 or 4 heads or else I would get an OOM error on my GPU's VRAM.

In order to reduce the first dimension (seq length) of the predictions, I used a mean layer right before the classification layer.

At the end, there are 3 code examples in order to show the predictions of some random sentences.
