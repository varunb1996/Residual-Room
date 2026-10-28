# sentence-transformers docs

# README.md

<!--- BADGES: START --->

[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/models?library=sentence-transformers)
[![GitHub - License](https://img.shields.io/github/license/huggingface/sentence-transformers?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sentence-transformers?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/sentence-transformers?logo=pypi&style=flat&color=orange)][#pypi-package]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=sentence-transformers)][#docs-package]

<!-- [![PyPI - Downloads](https://img.shields.io/pypi/dm/sentence-transformers?logo=pypi&style=flat&color=green)][#pypi-package] -->

<!--- BADGES: END --->

## Sentence Transformers: Embeddings, Retrieval, and Reranking

This framework provides an easy method to compute embeddings for accessing, using, and training state-of-the-art embedding and reranker models. It can be used to compute embeddings using Sentence Transformer models ([quickstart](https://sbert.net/docs/quickstart.html#sentence-transformer)), to calculate similarity scores using Cross-Encoder (a.k.a. reranker) models ([quickstart](https://sbert.net/docs/quickstart.html#cross-encoder)), to generate sparse embeddings using Sparse Encoder models ([quickstart](https://sbert.net/docs/quickstart.html#sparse-encoder)) or to compute token-level embeddings for ColBERT-style late-interaction retrieval using Multi-Vector Encoder models ([quickstart](https://sbert.net/docs/quickstart.html#multi-vector-encoder)). This unlocks a wide range of applications, including [semantic search](https://sbert.net/examples/applications/semantic-search/README.html), [semantic textual similarity](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html), and [paraphrase mining](https://sbert.net/examples/applications/paraphrase-mining/README.html).

A wide selection of over [15,000 pre-trained Sentence Transformers models](https://huggingface.co/models?library=sentence-transformers) are available for immediate use on 🤗 Hugging Face, including many of the state-of-the-art models from the [Massive Text Embeddings Benchmark (MTEB) leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Additionally, it is easy to train or finetune your own [embedding models](https://sbert.net/docs/sentence_transformer/training_overview.html), [reranker models](https://sbert.net/docs/cross_encoder/training_overview.html), [sparse encoder models](https://sbert.net/docs/sparse_encoder/training_overview.html) or [multi-vector encoder models](https://sbert.net/docs/multi_vector_encoder/training_overview.html) using Sentence Transformers, enabling you to create custom models for your specific use cases.

For the **full documentation**, see **[www.SBERT.net](https://www.sbert.net)**.

### Installation

We recommend **Python 3.10+**, **[PyTorch 2.2+](https://pytorch.org/get-started/locally/)**, and **[transformers v5.0+](https://github.com/huggingface/transformers)**.

```
pip install -U sentence-transformers
```

See [Installation](https://www.sbert.net/docs/installation.html) in the docs for uv, conda, source, and editable installs, CUDA setup, and extras (`[image]`, `[audio]`, `[video]`, `[train]`, `[onnx]`, `[openvino]`, `[dev]`).

### Getting Started

See [Quickstart](https://www.sbert.net/docs/quickstart.html) in our documentation.

Log in with `hf auth login` for higher Hugging Face Hub rate limits and access to private or gated models your account can read. See the [login instructions](https://www.sbert.net/docs/installation.html#log-in-to-hugging-face).

#### Embedding Models

First download a pretrained embedding a.k.a. Sentence Transformer model.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

Then provide some texts to the model.

```python
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# => (3, 384)
```

And that's already it. We now have numpy arrays with the embeddings, one for each text. We can use these to compute similarities.

```python
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
```

#### Reranker Models

First download a pretrained reranker a.k.a. Cross Encoder model.

```python
from sentence_transformers import CrossEncoder

# 1. Load a pretrained CrossEncoder model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
```

Then provide some texts to the model.

```python
# The texts for which to predict similarity scores
query = "How many people live in Berlin?"
passages = [
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
    "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
]

# 2a. predict scores for pairs of texts
scores = model.predict([(query, passage) for passage in passages])
print(scores)
# => [8.607139 5.506266 6.352977]
```

And we're good to go. You can also use [`model.rank`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.model.CrossEncoder.rank) to avoid having to perform the reranking manually:

```python
# 2b. Rank a list of passages for a query
ranks = model.rank(query, passages, return_documents=True)

print("Query:", query)
for rank in ranks:
    print(f"- #{rank['corpus_id']} ({rank['score']:.2f}): {rank['text']}")
"""
Query: How many people live in Berlin?
- #0 (8.61): Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.
- #2 (6.35): In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.
- #1 (5.51): Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.
"""
```

#### Sparse Encoder Models

First download a pretrained sparse embedding a.k.a. Sparse Encoder model.

```python

from sentence_transformers import SparseEncoder

# 1. Load a pretrained SparseEncoder model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate sparse embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 30522] - sparse representation with vocabulary size dimensions

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[   35.629,     9.154,     0.098],
#         [    9.154,    27.478,     0.019],
#         [    0.098,     0.019,    29.553]])

# 4. Check sparsity stats
stats = SparseEncoder.sparsity(embeddings)
print(f"Sparsity: {stats['sparsity_ratio']:.2%}")
# Sparsity: 99.84%
```

#### Multi-Vector Encoder Models

First download a pretrained multi-vector a.k.a. late-interaction (ColBERT-style) model.

```python
from sentence_transformers import MultiVectorEncoder

# 1. Load a pretrained MultiVectorEncoder model
model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

queries = ["What is the capital of France?"]
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
]

# 2. Encode queries and documents into sequences of token-level embeddings
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings[0].shape, document_embeddings[0].shape)
# (10, 128) (9, 128)  # one 128-dimensional vector per token

# 3. Score them with late interaction (MaxSim)
scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[9.6037, 9.4055]])
```

### Pre-Trained Models

We provide a large list of pretrained models for more than 100 languages. Some models are general purpose models, while others produce embeddings for specific use cases.

- [Pretrained Sentence Transformer (Embedding) Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
- [Pretrained Cross Encoder (Reranker) Models](https://sbert.net/docs/cross_encoder/pretrained_models.html)
- [Pretrained Sparse Encoder (Sparse Embeddings) Models](https://sbert.net/docs/sparse_encoder/pretrained_models.html)
- [Pretrained Multi-Vector Encoder (Late Interaction) Models](https://sbert.net/docs/multi_vector_encoder/pretrained_models.html)

### Training

> **Tip:** Using an AI coding agent (Claude Code, Codex, Cursor, Gemini CLI, ...)? Install the [`train-sentence-transformers`](skills/) Hugging Face Agent Skill via `hf skills add train-sentence-transformers [--claude] [--global]` and ask your agent to fine-tune a model on your data.

This framework allows you to fine-tune your own sentence embedding methods, so that you get task-specific sentence embeddings. You have various options to choose from in order to get perfect sentence embeddings for your specific task.

- Embedding Models
  - [Sentence Transformer > Training Overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html)
  - [Sentence Transformer > Training Examples](https://www.sbert.net/docs/sentence_transformer/training/examples.html) or [training examples on GitHub](https://github.com/huggingface/sentence-transformers/tree/main/examples/sentence_transformer/training).
- Reranker Models
  - [Cross Encoder > Training Overview](https://www.sbert.net/docs/cross_encoder/training_overview.html)
  - [Cross Encoder > Training Examples](https://www.sbert.net/docs/cross_encoder/training/examples.html) or [training examples on GitHub](https://github.com/huggingface/sentence-transformers/tree/main/examples/cross_encoder/training).
- Sparse Embedding Models
  - [Sparse Encoder > Training Overview](https://www.sbert.net/docs/sparse_encoder/training_overview.html)
  - [Sparse Encoder > Training Examples](https://www.sbert.net/docs/sparse_encoder/training/examples.html) or [training examples on GitHub](https://github.com/huggingface/sentence-transformers/tree/main/examples/sparse_encoder/training).
- Multi-Vector (Late Interaction) Models
  - [Multi-Vector Encoder > Training Overview](https://www.sbert.net/docs/multi_vector_encoder/training_overview.html)
  - [Training examples on GitHub](https://github.com/huggingface/sentence-transformers/tree/main/examples/multi_vector_encoder/training).

Some highlights across the different types of training are:

- Support of various transformer networks including BERT, RoBERTa, XLM-R, DistilBERT, Electra, BART, ...
- Multilingual and multi-task learning
- Evaluation during training to find optimal model
- [20+ loss functions](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html) for embedding models, [10+ loss functions](https://www.sbert.net/docs/package_reference/cross_encoder/losses.html) for reranker models and [10+ loss functions](https://www.sbert.net/docs/package_reference/sparse_encoder/losses.html) for sparse embedding models, allowing you to tune models specifically for semantic search, paraphrase mining, semantic similarity comparison, clustering, triplet loss, contrastive loss, etc.

### Companion Blog Posts

The following Hugging Face blog posts complement this documentation with narrative walkthroughs and full training examples:

**Training guides:**

- [Training and Finetuning Embedding Models](https://huggingface.co/blog/train-sentence-transformers): end-to-end training of bi-encoder embedding models.
- [Training and Finetuning Reranker Models](https://huggingface.co/blog/train-reranker): training Cross Encoder models for the second stage of retrieve-and-rerank pipelines.
- [Training and Finetuning Sparse Embedding Models](https://huggingface.co/blog/train-sparse-encoder): training SPLADE and other sparse encoders.

**Multimodal:**

- [Multimodal Embedding & Reranker Models](https://huggingface.co/blog/multimodal-sentence-transformers): using text, image, audio, and video models through a single API.
- [Training and Finetuning Multimodal Embedding & Reranker Models](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal models, with a Visual Document Retrieval walkthrough.

**Efficiency techniques:**

- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Train 400x faster Static Embedding Models](https://huggingface.co/blog/static-embeddings): CPU-friendly embedding models without attention.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.

### Application Examples

You can use this framework for:

- **Computing Sentence Embeddings**

  - [Dense Embeddings](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html)
  - [Sparse Embeddings](https://www.sbert.net/examples/sparse_encoder/applications/computing_embeddings/README.html)

- **Semantic Textual Similarity**

  - [Dense STS](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)
  - [Sparse STS](https://www.sbert.net/examples/sparse_encoder/applications/semantic_textual_similarity/README.html)

- **Semantic Search**

  - [Dense Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
  - [Sparse Search](https://www.sbert.net/examples/sparse_encoder/applications/semantic_search/README.html)

- **Retrieve & Re-Rank**

  - [Dense only Retrieval](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
  - [Sparse/Dense/Hybrid Retrieval](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

- [Clustering](https://www.sbert.net/examples/sentence_transformer/applications/clustering/README.html)

- [Paraphrase Mining](https://www.sbert.net/examples/sentence_transformer/applications/paraphrase-mining/README.html)

- [Translated Sentence Mining](https://www.sbert.net/examples/sentence_transformer/applications/parallel-sentence-mining/README.html)

- [Multilingual Image Search, Clustering & Duplicate Detection](https://www.sbert.net/examples/sentence_transformer/applications/image-search/README.html)

and many more use-cases.

For all examples, see [examples/sentence_transformer/applications](https://github.com/huggingface/sentence-transformers/tree/main/examples/sentence_transformer/applications).

### Contributing

See the [contributor guidelines](CONTRIBUTING.md) for contribution expectations, AI usage guidelines, and development setup.

### Citing & Authors

If you find this repository helpful, feel free to cite our publication [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://huggingface.co/papers/1908.10084):

```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

If you use one of the multilingual models, feel free to cite our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://huggingface.co/papers/2004.09813):

```bibtex
@inproceedings{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2004.09813",
}
```

Please have a look at [Publications](https://www.sbert.net/docs/publications.html) for our different publications that are integrated into SentenceTransformers.

#### Maintainers

Maintainer: [Tom Aarsen](https://github.com/tomaarsen), 🤗 Hugging Face

Don't hesitate to open an issue if something is broken (and it shouldn't be) or if you have further questions.

---

This project was originally developed by the [Ubiquitous Knowledge Processing (UKP) Lab](https://www.ukp.tu-darmstadt.de/) at TU Darmstadt. We're grateful for their foundational work and continued contributions to the field.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

[#docs-package]: https://www.sbert.net/
[#github-license]: https://github.com/huggingface/sentence-transformers/blob/main/LICENSE
[#pypi-package]: https://pypi.org/project/sentence-transformers/

# docs/sentence_transformer/usage/semantic_textual_similarity.rst

## Semantic Textual Similarity

For Semantic Textual Similarity (STS), we want to produce embeddings for all texts involved and calculate the similarities between them. The text pairs with the highest similarity score are most semantically similar. See also the `Computing Embeddings <../../../examples/sentence_transformer/applications/computing-embeddings/README.html>`_ documentation for more advanced details on getting embedding scores.

.. sidebar:: Documentation

   1. :class:`SentenceTransformer <sentence_transformers.sentence_transformer.model.SentenceTransformer>`
   2. :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`
   3. :meth:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>`

::

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Two lists of sentences
    sentences1 = [
        "The new movie is awesome",
        "The cat sits outside",
        "A man is playing guitar",
    ]

    sentences2 = [
        "The dog plays in the garden",
        "The new movie is so great",
        "A woman watches TV",
    ]

    # Compute embeddings for both lists
    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    # Compute cosine similarities
    similarities = model.similarity(embeddings1, embeddings2)

    # Output the pairs with their score
    for idx_i, sentence1 in enumerate(sentences1):
        print(sentence1)
        for idx_j, sentence2 in enumerate(sentences2):
            print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")

.. code-block:: text
    :emphasize-lines: 3

    The new movie is awesome
    - The dog plays in the garden   : 0.0543
    - The new movie is so great     : 0.8939
    - A woman watches TV            : -0.0502
    The cat sits outside
    - The dog plays in the garden   : 0.2838
    - The new movie is so great     : -0.0029
    - A woman watches TV            : 0.1310
    A man is playing guitar
    - The dog plays in the garden   : 0.2277
    - The new movie is so great     : -0.0136
    - A woman watches TV            : -0.0327

In this example, the :meth:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>` method returns a 3x3 matrix with the respective cosine similarity scores for all possible pairs between ``embeddings1`` and ``embeddings2``.

### Similarity Calculation

The similarity metric that is used is stored in the SentenceTransformer instance under :attr:`SentenceTransformer.similarity_fn_name <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity_fn_name>`. Valid options are:

- ``SimilarityFunction.COSINE`` (a.k.a `"cosine"`): Cosine Similarity (**default**)
- ``SimilarityFunction.DOT_PRODUCT`` (a.k.a `"dot"`): Dot Product
- ``SimilarityFunction.EUCLIDEAN`` (a.k.a `"euclidean"`): Negative Euclidean Distance
- ``SimilarityFunction.MANHATTAN`` (a.k.a. `"manhattan"`): Negative Manhattan Distance

This value can be changed in a handful of ways:

1. By initializing the SentenceTransformer instance with the desired similarity function::

    from sentence_transformers import SentenceTransformer, SimilarityFunction

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.DOT_PRODUCT)

2. By setting the value directly on the SentenceTransformer instance::

    from sentence_transformers import SentenceTransformer, SimilarityFunction

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT

3. By setting the value under the ``"similarity_fn_name"`` key in the ``config_sentence_transformers.json`` file of a saved model. When you save a Sentence Transformer model, this value will be automatically saved as well.

Sentence Transformers implements two methods to calculate the similarity between embeddings:

- :meth:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>`: Calculates the similarity between all pairs of embeddings.
- :meth:`SentenceTransformer.similarity_pairwise <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity_pairwise>`: Calculates the similarity between embeddings in a pairwise fashion.

::

    from sentence_transformers import SentenceTransformer, SimilarityFunction

    # Load a pretrained Sentence Transformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Embed some sentences
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    embeddings = model.encode(sentences)

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    # tensor([[1.0000, 0.6660, 0.1046],
    #         [0.6660, 1.0000, 0.1411],
    #         [0.1046, 0.1411, 1.0000]])

    # Change the similarity function to Manhattan distance
    model.similarity_fn_name = SimilarityFunction.MANHATTAN
    print(model.similarity_fn_name)
    # => "manhattan"

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    # tensor([[ -0.0000, -12.6269, -20.2167],
    #         [-12.6269,  -0.0000, -20.1288],
    #         [-20.2167, -20.1288,  -0.0000]])

.. note::

   If a Sentence Transformer instance ends with a :class:`~sentence_transformers.sentence_transformer.modules.Normalize` module, then it is sensible to choose the "dot" metric instead of "cosine".

   Dot product on normalized embeddings is equivalent to cosine similarity, but "cosine" will re-normalize the embeddings again. As a result, the "dot" metric will be faster than "cosine".

If you want find the highest scoring pairs in a long list of sentences, have a look at `Paraphrase Mining <../../../examples/sentence_transformer/applications/paraphrase-mining/README.html>`_.

# examples/sentence_transformer/applications/semantic-search/README.md

## Semantic Search

Semantic search seeks to improve search accuracy by understanding the semantic meaning of the search query and the corpus to search over. Semantic search can also perform well given synonyms, abbreviations, and misspellings, unlike keyword search engines that can only find documents based on lexical matches.

### Background

The idea behind semantic search is to embed all entries in your corpus, whether they be sentences, paragraphs, or documents, into a vector space. At search time, the query is embedded into the same vector space and the closest embeddings from your corpus are found. These entries should have a high semantic similarity with the query.

![SemanticSearch](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/SemanticSearch.png)

### Symmetric vs. Asymmetric Semantic Search

A **critical distinction** for your setup is *symmetric* vs. *asymmetric semantic search*:

- For **symmetric semantic search** your query and the entries in your corpus are of about the same length and have the same amount of content. An example would be searching for similar questions: Your query could for example be *"How to learn Python online?"* and you want to find an entry like *"How to learn Python on the web?"*. For symmetric tasks, you could potentially flip the query and the entries in your corpus.
  - Related training example: [Quora Duplicate Questions](../../training/quora_duplicate_questions/README.md).
  - Suitable models: [Pre-Trained Sentence Embedding Models](../../../../docs/sentence_transformer/pretrained_models.md)
- For **asymmetric semantic search**, you usually have a **short query** (like a question or some keywords) and you want to find a longer paragraph answering the query. An example would be a query like *"What is Python"* and you want to find the paragraph *"Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy ..."*. For asymmetric tasks, flipping the query and the entries in your corpus usually does not make sense.
  - Related training example: [MS MARCO](../../training/ms_marco/README.md)
  - Suitable models: [Pre-Trained MS MARCO Models](../../../../docs/pretrained-models/msmarco-v5.md)

It is critical **that you choose the right model** for your type of task.

```{eval-rst}
.. tip::

    For asymmetric semantic search, you are recommended to use :meth:`SentenceTransformer.encode_query <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query>` to encode your queries and :meth:`SentenceTransformer.encode_document <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document>` to encode your corpus. 
    
    The more general :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>` method differs in two ways from :meth:`SentenceTransformer.encode_query <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query>` and :meth:`SentenceTransformer.encode_document <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document>`:

    1. If no ``prompt_name`` or ``prompt`` is provided, it uses a predefined "query" or "document" prompt, if specified in the model's ``prompts`` dictionary.
    2. It sets the ``task`` to "document". If the model has a :class:`~sentence_transformers.base.modules.Router` module, it will use the "query" or "document" task type to route the input through the appropriate submodules.

    Note that :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>` is the most general method and can be used for any task, including Information Retrieval, and that if the model was not trained with predefined prompts and/or task types, then all three methods will return identical embeddings.

```

### Manual Implementation

```{eval-rst}
For small corpora (up to about 1 million entries), we can perform semantic search with a manual implementation by computing the embeddings for the corpus with :meth:`SentenceTransformer.encode_document <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document>` as well as for our query with :meth:`SentenceTransformer.encode_query <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query>`, and then calculating the `semantic textual similarity <../../../../docs/sentence_transformer/usage/semantic_textual_similarity.html>`_ using :func:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>`.
```

For a simple example, see [semantic_search.py](semantic_search.py):

```{eval-rst}

.. sidebar:: Output

   .. code-block:: text

        Query: How do artificial neural networks work?
        Top 5 most similar sentences in corpus:
        (Score: 0.5926) Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.
        (Score: 0.5288) Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.
        (Score: 0.4647) Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.
        (Score: 0.1381) Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments.
        (Score: 0.0912) Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground.

        Query: What technology is used for modern space exploration?
        Top 5 most similar sentences in corpus:
        (Score: 0.3754) Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments.
        (Score: 0.3669) SpaceX's Starship is designed to be a fully reusable transportation system capable of carrying humans to Mars and beyond.
        (Score: 0.3452) The James Webb Space Telescope is the largest optical telescope in space, designed to conduct infrared astronomy.
        (Score: 0.2625) Renewable energy sources include solar, wind, hydro, and geothermal power that naturally replenish over time.
        (Score: 0.2275) Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground.

        Query: How can we address climate change challenges?
        Top 5 most similar sentences in corpus:
        (Score: 0.3760) Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period due to human activities.
        (Score: 0.3144) Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground.
        (Score: 0.2948) Renewable energy sources include solar, wind, hydro, and geothermal power that naturally replenish over time.
        (Score: 0.0420) Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.
        (Score: 0.0411) Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.

.. literalinclude:: semantic_search.py
```

### Optimized Implementation

```{eval-rst}
Instead of implementing semantic search by yourself, you can use the :func:`~sentence_transformers.util.semantic_search` function.
```

The function accepts the following parameters:

```{eval-rst}
.. autofunction:: sentence_transformers.util.semantic_search
```

By default, up to 100 queries are processed in parallel. Further, the corpus is chunked into set of up to 500k entries. You can increase `query_chunk_size` and `corpus_chunk_size`, which leads to increased speed for large corpora, but also increases the memory requirement.

### Speed Optimization

```{eval-rst}
To get the optimal speed for the :func:`~sentence_transformers.util.semantic_search` method, it is advisable to have the ``query_embeddings`` as well as the ``corpus_embeddings`` on the same GPU-device. This significantly boost the performance. Further, we can normalize the corpus embeddings so that each corpus embeddings is of length 1. In that case, we can use dot-product for computing scores.

.. code-block:: python

    from sentence_transformers.util import dot_score, normalize_embeddings, semantic_search

    corpus_embeddings = corpus_embeddings.to("cuda")
    corpus_embeddings = normalize_embeddings(corpus_embeddings)

    query_embeddings = query_embeddings.to("cuda")
    query_embeddings = normalize_embeddings(query_embeddings)
    hits = semantic_search(query_embeddings, corpus_embeddings, score_function=dot_score)
```

### Elasticsearch

[Elasticsearch](https://www.elastic.co/elasticsearch/) has the possibility to [index dense vectors](https://www.elastic.co/what-is/vector-search) and to use them for document scoring. We can easily index embedding vectors, store other data alongside our vectors and, most importantly, efficiently retrieve relevant entries using [approximate nearest neighbor search](https://www.elastic.co/blog/introducing-approximate-nearest-neighbor-search-in-elasticsearch-8-0) (HNSW, see also below) on the embeddings.

For further details, see [semantic_search_quora_elasticsearch.py](semantic_search_quora_elasticsearch.py).

### OpenSearch

[OpenSearch](https://opensearch.org/) is a community-driven, open-source search engine that supports vector search capabilities. It allows you to index dense vectors and perform efficient similarity search using approximate nearest neighbor algorithms. OpenSearch can be used to implement both traditional keyword-based search (BM25) and semantic search, making it possible to compare and combine both approaches.

For an example implementation, see [semantic_search_nq_opensearch.py](semantic_search_nq_opensearch.py), which shows how to use OpenSearch with the Natural Questions dataset, demonstrating both semantic search and BM25 search capabilities.

### Approximate Nearest Neighbor

```{eval-rst}
Searching a large corpus with millions of embeddings can be time-consuming if exact nearest neighbor search is used (like it is used by :func:`~sentence_transformers.util.semantic_search`).
```

In that case, Approximate Nearest Neighbor (ANN) can be helpful. Here, the data is partitioned into smaller fractions of similar embeddings. This index can be searched efficiently and the embeddings with the highest similarity (the nearest neighbors) can be retrieved within milliseconds, even if you have millions of vectors. However, the results are not necessarily exact. It is possible that some vectors with high similarity will be missed.

For all ANN methods, there are usually one or more parameters to tune that determine the recall-speed trade-off. If you want the highest speed, you have a high chance of missing hits. If you want high recall, the search speed decreases.

Three popular libraries for approximate nearest neighbor are [Annoy](https://github.com/spotify/annoy), [FAISS](https://github.com/facebookresearch/faiss), and [hnswlib](https://github.com/nmslib/hnswlib/).

Examples:

- [semantic_search_quora_hnswlib.py](semantic_search_quora_hnswlib.py)
- [semantic_search_quora_annoy.py](semantic_search_quora_annoy.py)
- [semantic_search_quora_faiss.py](semantic_search_quora_faiss.py)

### Retrieve & Re-Rank

For complex semantic search scenarios, a two-stage retrieve & re-rank pipeline is advisable:
![InformationRetrieval](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/InformationRetrieval.png)

For further details, see [Retrieve & Re-rank](../retrieve_rerank/README.md).

### Examples

We list a handful of common use cases:

#### Similar Questions Retrieval

[semantic_search_quora_pytorch.py](semantic_search_quora_pytorch.py) [ [Colab version](https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing) ] shows an example based on the [Quora duplicate questions](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) dataset. The user can enter a question, and the code retrieves the most similar questions from the dataset using `semantic_search`. As model, we use [distilbert-multilingual-nli-stsb-quora-ranking](https://huggingface.co/sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking), which was trained to identify similar questions and supports 50+ languages. Hence, the user can input the question in any of the 50+ languages. This is a **symmetric search task**, as the search queries have the same length and content as the questions in the corpus.

#### Similar Publication Retrieval

[semantic_search_publications.py](semantic_search_publications.py) [ [Colab version](https://colab.research.google.com/drive/12hfBveGHRsxhPIUMmJYrll2lFU4fOX06?usp=sharing) ] shows an example how to find similar scientific publications. As corpus, we use all publications that have been presented at the EMNLP 2016 - 2018 conferences. As search query, we input the title and abstract of more recent publications and find related publications from our corpus. We use the [SPECTER](https://huggingface.co/sentence-transformers/allenai-specter) model. This is a **symmetric search task**, as the paper in the corpus consists of title & abstract and we search for title & abstract.

#### Question & Answer Retrieval

[semantic_search_wikipedia_qa.py](semantic_search_wikipedia_qa.py) [ [Colab Version](https://colab.research.google.com/drive/11GunvCqJuebfeTlgbJWkIMT0xJH6PWF1?usp=sharing) ]: This example uses a model that was trained on the [Natural Questions dataset](https://huggingface.co/datasets/sentence-transformers/natural-questions). It consists of about 100k real Google search queries, together with an annotated passage from Wikipedia that provides the answer. It is an example of an **asymmetric search task**. As corpus, we use the smaller [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) so that it fits easily into memory.

[retrieve_rerank_simple_wikipedia.ipynb](../retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) [ [Colab Version](https://colab.research.google.com/github/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) ]: This script uses the [Retrieve & Re-rank](../retrieve_rerank/README.md) strategy and is an example for an **asymmetric search task**. We split all Wikipedia articles into paragraphs and encode them with a bi-encoder. If a new query / question is entered, it is encoded by the same bi-encoder and the paragraphs with the highest cosine-similarity are retrieved. Next, the retrieved candidates are scored by a Cross-Encoder re-ranker and the 5 passages with the highest score from the Cross-Encoder are presented to the user. We use models that were trained on the [MS Marco Passage Reranking](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset, a dataset with about 500k real queries from Bing search.

# examples/sentence_transformer/applications/computing-embeddings/README.rst

## Computing Embeddings

Once you have `installed <../../../../docs/installation.html>`_ Sentence Transformers, you can easily use Sentence Transformer models:

.. sidebar:: Documentation

   1. :class:`SentenceTransformer <sentence_transformers.sentence_transformer.model.SentenceTransformer>`
   2. :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`
   3. :meth:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>`

::

   from sentence_transformers import SentenceTransformer

   # 1. Load a pretrained Sentence Transformer model
   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

   # The sentences to encode
   sentences = [
       "The weather is lovely today.",
       "It's so sunny outside!",
       "He drove to the stadium.",
   ]

   # 2. Calculate embeddings by calling model.encode()
   embeddings = model.encode(sentences)
   print(embeddings.shape)
   # [3, 384]

   # 3. Calculate the embedding similarities
   similarities = model.similarity(embeddings, embeddings)
   print(similarities)
   # tensor([[1.0000, 0.6660, 0.1046],
   #         [0.6660, 1.0000, 0.1411],
   #         [0.1046, 0.1411, 1.0000]])

.. note::
   Even though we talk about sentence embeddings, you can use Sentence Transformers for shorter phrases as well as for longer texts with multiple sentences. See :ref:`input-sequence-length` for notes on embeddings for longer texts.


### Initializing a Sentence Transformer Model

The first step is to load a pretrained Sentence Transformer model. You can use any of the models from the `Pretrained Models <../../../../docs/sentence_transformer/pretrained_models.html>`_ or a local model. See also :class:`~sentence_transformers.sentence_transformer.model.SentenceTransformer` for information on parameters.

::

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
   # Alternatively, you can pass a path to a local model directory:
   model = SentenceTransformer("output/models/mpnet-base-finetuned-all-nli")

The model will automatically be placed on the most performant available device, e.g. ``cuda`` or ``mps`` if available. You can also specify the device explicitly:

::

   model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

### Calculating Embeddings

The method to calculate embeddings is :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`.


### Prompt Templates

Some models require using specific text *prompts* to achieve optimal performance. For example, with `intfloat/multilingual-e5-large <https://huggingface.co/intfloat/multilingual-e5-large>`_ you should prefix all queries with ``"query: "`` and all passages with ``"passage: "``. Another example is `BAAI/bge-large-en-v1.5 <https://huggingface.co/BAAI/bge-large-en-v1.5>`_, which performs best for retrieval when the input texts are prefixed with ``"Represent this sentence for searching relevant passages: "``. 

Sentence Transformer models can be initialized with ``prompts`` and ``default_prompt_name`` parameters:

- ``prompts`` is an optional argument that accepts a dictionary of prompts with prompt names to prompt texts. The prompt will be prepended to the input text during inference. For example::

    model = SentenceTransformer(
        "intfloat/multilingual-e5-large",
        prompts={
            "classification": "Classify the following text: ",
            "retrieval": "Retrieve semantically similar text: ",
            "clustering": "Identify the topic or theme based on the text: ",
        },
    )
    # or
    model.prompts = {
        "classification": "Classify the following text: ",
        "retrieval": "Retrieve semantically similar text: ",
        "clustering": "Identify the topic or theme based on the text: ",
    }

- ``default_prompt_name`` is an optional argument that determines the default prompt to be used. It has to correspond with a prompt name from ``prompts``. If ``None``, then no prompt is used by default. For example::

    model = SentenceTransformer(
        "intfloat/multilingual-e5-large",
        prompts={
            "classification": "Classify the following text: ",
            "retrieval": "Retrieve semantically similar text: ",
            "clustering": "Identify the topic or theme based on the text: ",
        },
        default_prompt_name="retrieval",
    )
    # or
    model.default_prompt_name="retrieval"

Both of these parameters can also be specified in the ``config_sentence_transformers.json`` file of a saved model. That way, you won't have to specify these options manually when loading. When you save a Sentence Transformer model, these options will be automatically saved as well.

During inference, prompts can be applied in a few different ways. All of these scenarios result in identical texts being embedded:

1. Explicitly using the ``prompt`` option in ``SentenceTransformer.encode``::

    embeddings = model.encode("How to bake a strawberry cake", prompt="Retrieve semantically similar text: ")

2. Explicitly using the ``prompt_name`` option in ``SentenceTransformer.encode`` by relying on the prompts loaded from a) initialization or b) the model config::

    embeddings = model.encode("How to bake a strawberry cake", prompt_name="retrieval")

3. If ``prompt`` nor ``prompt_name`` are specified in ``SentenceTransformer.encode``, then the prompt specified by ``default_prompt_name`` will be applied. If it is ``None``, then no prompt will be applied::

    embeddings = model.encode("How to bake a strawberry cake")

.. _input-sequence-length:

### Input Sequence Length

For transformer models like BERT, RoBERTa, DistilBERT etc., the runtime and memory requirement grows quadratic with the input length. This limits transformers to inputs of certain lengths. A common value for BERT-based models are 512 tokens, which corresponds to about 300-400 words (for English).

Each model has a maximum sequence length under ``model.max_seq_length``, which is the maximal number of tokens that can be processed. Longer texts will be truncated to the first ``model.max_seq_length`` tokens::

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Max Sequence Length:", model.max_seq_length)
    # => Max Sequence Length: 256

    # Change the length to 200
    model.max_seq_length = 200

    print("Max Sequence Length:", model.max_seq_length)
    # => Max Sequence Length: 200

.. note::

   You cannot increase the length higher than what is maximally supported by the respective transformer model. Also note that if a model was trained on short texts, the representations for long texts might not be that good.

### Multi-Process / Multi-GPU Encoding

You can encode input texts with more than one GPU (or with multiple processes on a CPU machine). It tends to help significantly with large datasets, but the overhead of starting multiple processes can be significant for smaller datasets.
For an example, see: `computing_embeddings_multi_gpu.py <https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/computing-embeddings/computing_embeddings_multi_gpu.py>`_.
 
You can use :meth:`SentenceTransformer.encode() <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>` (or :meth:`SentenceTransformer.encode_query() <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query>` or :meth:`SentenceTransformer.encode_document() <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document>`) with either:

- The ``device`` parameter, which can be set to e.g. ``"cuda:0"`` or ``"cpu"`` for single-process computations, but also a list of devices for multi-process or multi-gpu computations, e.g. ``["cuda:0", "cuda:1"]`` or ``["cpu", "cpu", "cpu", "cpu"]``::

        from sentence_transformers import SentenceTransformer

        def main():
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            # Encode with multiple GPUs
            embeddings = model.encode(
                inputs,
                device=["cuda:0", "cuda:1"]  # or ["cpu", "cpu", "cpu", "cpu"]
            )

        if __name__ == "__main__":
            main()

- The ``pool`` parameter can be provided, after calling :meth:`SentenceTransformer.start_multi_process_pool() <sentence_transformers.sentence_transformer.model.SentenceTransformer.start_multi_process_pool>` with a list of devices, e.g. ``["cuda:0", "cuda:1"]`` or ``["cpu", "cpu", "cpu", "cpu"]``. The benefit of this is that the pool can be reused for multiple calls to :meth:`SentenceTransformer.encode() <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`, which is considerably more efficient than starting a new pool for each call::

        from sentence_transformers import SentenceTransformer

        def main():
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            # Start a multi-process pool with multiple GPUs
            pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1"])
            # Encode with multiple GPUs
            embeddings = model.encode(inputs, pool=pool)
            # Don't forget to stop the pool after usage
            model.stop_multi_process_pool(pool)
        
        if __name__ == "__main__":
            main()

Additionally, you can use the ``chunk_size`` parameter to control the size of the chunks sent to each process. This differs from the ``batch_size`` parameter. For example, with a ``chunk_size=1000`` and a ``batch_size=32``, the input texts will be split into chunks of 1000 texts, and each chunk will be sent to a process and embedded in batches of 32 texts at a time. This can help with memory management and performance, especially for large datasets.
