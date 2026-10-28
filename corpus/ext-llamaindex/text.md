# LlamaIndex docs

# docs/src/content/docs/framework/understanding/rag/index.mdx

If you haven't, [install
  LlamaIndex](/python/framework/getting_started/installation) and complete the
  [starter tutorial](/python/framework/getting_started/starter_example) before
  you read this. It will help ground these steps in your experience.

LLMs are trained on enormous bodies of data but they aren't trained on **your** data. Retrieval-Augmented Generation (RAG) solves this problem by adding your data to the data LLMs already have access to. You will see references to RAG frequently in this documentation. Query engines, chat engines and agents often use RAG to complete their tasks.

In RAG, your data is loaded and prepared for queries or "indexed". User queries act on the index, which filters your data down to the most relevant context. This context and your query then go to the LLM along with a prompt, and the LLM provides a response.

Even if what you're building is a chatbot or an agent, you'll want to know RAG techniques for getting data into your application.

![](./../../_static/getting_started/basic_rag.png)

#### Stages within RAG

There are five key stages within RAG, which in turn will be a part of most larger applications you build. These are:

- **Loading**: this refers to getting your data from where it lives -- whether it's text files, PDFs, another website, a database, or an API -- into your workflow. LlamaIndex provides [hundreds of connectors](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers) to choose from.

- **Indexing**: this means creating a data structure that allows for querying the data. For LLMs this nearly always means creating `vector embeddings`, numerical representations of the meaning of your data, as well as numerous other metadata strategies to make it easy to accurately find contextually relevant data.

- **Storing**: once your data is indexed you will almost always want to store your index, as well as other metadata, to avoid having to re-index it.

- **Querying**: for any given indexing strategy there are many ways you can utilize LLMs and LlamaIndex data structures to query, including sub-queries, multi-step queries and hybrid strategies.

- **Evaluation**: a critical step in any flow is checking how effective it is relative to other strategies, or when you make changes. Evaluation provides objective measures of how accurate, faithful and fast your responses to queries are.

![](./../../_static/getting_started/stages.png)

#### Important concepts within RAG

There are also some terms you'll encounter that refer to steps within each of these stages.

##### Loading stage

[**Nodes and Documents**](/python/framework/module_guides/loading/documents_and_nodes): A `Document` is a container around any data source - for instance, a PDF, an API output, or retrieve data from a database. A `Node` is the atomic unit of data in LlamaIndex and represents a "chunk" of a source `Document`. Nodes have metadata that relate them to the document they are in and to other nodes.

[**Connectors**](/python/framework/module_guides/loading/connector):
A data connector (often called a `Reader`) ingests data from different data sources and data formats into `Documents` and `Nodes`.

##### Indexing Stage

[**Indexes**](/python/framework/module_guides/indexing):
Once you've ingested your data, LlamaIndex will help you index the data into a structure that's easy to retrieve. This usually involves generating `vector embeddings` which are stored in a specialized database called a `vector store`. Indexes can also store a variety of metadata about your data.

[**Embeddings**](/python/framework/module_guides/models/embeddings): LLMs generate numerical representations of data called `embeddings`. When filtering your data for relevance, LlamaIndex will convert queries into embeddings, and your vector store will find data that is numerically similar to the embedding of your query.

##### Querying Stage

[**Retrievers**](/python/framework/module_guides/querying/retriever):
A retriever defines how to efficiently retrieve relevant context from an index when given a query. Your retrieval strategy is key to the relevancy of the data retrieved and the efficiency with which it's done.

[**Routers**](/python/framework/module_guides/querying/router):
A router determines which retriever will be used to retrieve relevant context from the knowledge base. More specifically, the `RouterRetriever` class, is responsible for selecting one or multiple candidate retrievers to execute a query. They use a selector to choose the best option based on each candidate's metadata and the query.

[**Node Postprocessors**](/python/framework/module_guides/querying/node_postprocessors):
A node postprocessor takes in a set of retrieved nodes and applies transformations, filtering, or re-ranking logic to them.

[**Response Synthesizers**](/python/framework/module_guides/querying/response_synthesizers):
A response synthesizer generates a response from an LLM, using a user query and a given set of retrieved text chunks.

# docs/src/content/docs/framework/getting_started/concepts.mdx

This is a quick guide to the high-level concepts you'll encounter frequently when building LLM applications.

### Large Language Models (LLMs)

LLMs are the fundamental innovation that launched LlamaIndex. They are an artificial intelligence (AI) computer system that can understand, generate, and manipulate natural language, including answering questions based on their training data or data provided to them at query time. You can [learn more about using LLMs](/python/framework/understanding/using_llms).

### Agentic Applications

When an LLM is used within an application, it is often used to make decisions, take actions, and/or interact with the world. This is the core definition of an **agentic application**.

While the definition of an agentic application is broad, there are several key characteristics that define an agentic application:

- **LLM Augmentation**: The LLM is augmented with tools (i.e. arbitrary callable functions in code), memory, and/or dynamic prompts.
- **Prompt Chaining**: Several LLM calls are used that build on each other, with the output of one LLM call being used as the input to the next.
- **Routing**: The LLM is used to route the application to the next appropriate step or state in the application.
- **Parallelism**: The application can perform multiple steps or actions in parallel.
- **Orchestration**: A hierarchical structure of LLMs is used to orchestrate lower-level actions and LLMs.
- **Reflection**: The LLM is used to reflect and validate outputs of previous steps or LLM calls, which can be used to guide the application to the next appropriate step or state.

In LlamaIndex, you can build agentic applications by using the `Workflow` class to orchestrate a sequence of steps and LLMs. You can [learn more about workflows](/python/llamaagents/workflows).

### Agents

We define an agent as a specific instance of an "agentic application". An agent is a piece of software that semi-autonomously performs tasks by combining LLMs with other tools and memory, orchestrated in a reasoning loop that decides which tool to use next (if any).

What this means in practice, is something like:

- An agent receives a user message
- The agent uses an LLM to determine the next appropriate action to take using the previous chat history, tools, and the latest user message
- The agent may invoke one or more tools to assist in the users request
- If tools are used, the agent will then interpret the tool outputs and use them to inform the next action
- Once the agent stops taking actions, it returns the final output to the user

You can [learn more about agents](/python/framework/understanding/agent).

### Retrieval Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a core technique for building data-backed LLM applications with LlamaIndex. It allows LLMs to answer questions about your private data by providing it to the LLM at query time, rather than training the LLM on your data. To avoid sending **all** of your data to the LLM every time, RAG indexes your data and selectively sends only the relevant parts along with your query. You can [learn more about RAG](/python/framework/understanding/rag).

### Use cases

There are endless use cases for data-backed LLM applications but they can be roughly grouped into five categories:

[**Agents**](/python/framework/module_guides/deploying/agents):
An agent is an automated decision-maker powered by an LLM that interacts with the world via a set of [tools](/python/framework/module_guides/deploying/agents/tools). Agents can take an arbitrary number of steps to complete a given task, dynamically deciding on the best course of action rather than following pre-determined steps. This gives it additional flexibility to tackle more complex tasks.

[**Workflows**](/python/framework/llamaagents/workflows):
A Workflow in LlamaIndex is a specific event-driven abstraction that allows you to orchestrate a sequence of steps and LLMs calls. Workflows can be used to implement any agentic application, and are a core component of LlamaIndex.

[**Structured Data Extraction**](/python/framework/use_cases/extraction)
Pydantic extractors allow you to specify a precise data structure to extract from your data and use LLMs to fill in the missing pieces in a type-safe way. This is useful for extracting structured data from unstructured sources like PDFs, websites, and more, and is key to automating workflows.

[**Query Engines**](/python/framework/module_guides/deploying/query_engine):
A query engine is an end-to-end flow that allows you to ask questions over your data. It takes in a natural language query, and returns a response, along with reference context retrieved and passed to the LLM.

[**Chat Engines**](/python/framework/module_guides/deploying/chat_engines):
A chat engine is an end-to-end flow for having a conversation with your data (multiple back-and-forth instead of a single question-and-answer).

  * Tell me how to [customize things](/python/framework/getting_started/faq) *
  Continue learning with our [understanding
  LlamaIndex](/python/framework/understanding) guide * Ready to dig deep? Check
  out the [component guides](/python/framework/module_guides)

# docs/src/content/docs/framework/optimizing/production_rag.md

---
title: Building Performant RAG Applications for Production
---

Prototyping a RAG application is easy, but making it performant, robust, and scalable to a large knowledge corpus is hard.

This guide contains a variety of tips and tricks to improve the performance of your RAG workflow. We first outline
some general techniques - they are loosely ordered in terms of most straightforward to most challenging.
We then dive a bit more deeply into each technique, the use cases that it solves,
and how to implement it with LlamaIndex!

The end goal is to optimize your retrieval and generation performance to answer more
queries over more complex datasets accurately and without hallucinations.

### General Techniques for Building Production-Grade RAG

Here are some top Considerations for Building Production-Grade RAG

- Decoupling chunks used for retrieval vs. chunks used for synthesis
- Structured Retrieval for Larger Document Sets
- Dynamically Retrieve Chunks Depending on your Task
- Reranking retrieved nodes for precision
- Optimize context embeddings

We discussed this and more during our [Production RAG Webinar](https://www.youtube.com/watch?v=Zj5RCweUHIk).
Check out [this Tweet thread](https://twitter.com/jerryjliu0/status/1692931028963221929?s=20) for more synthesized details.

### Decoupling Chunks Used for Retrieval vs. Chunks Used for Synthesis

A key technique for better retrieval is to decouple chunks used for retrieval with those that are used for synthesis.

![](./../_static/production_rag/decouple_chunks.png)

##### Motivation

The optimal chunk representation for retrieval might be different than the optimal consideration used for synthesis.
For instance, a raw text chunk may contain needed details for the LLM to synthesize a more detailed answer given a query. However, it
may contain filler words/info that may bias the embedding representation, or it may lack global context and not be retrieved at all
when a relevant query comes in.

##### Key Techniques

There’s two main ways to take advantage of this idea:

**1. Embed a document summary, which links to chunks associated with the document.**

This can help retrieve relevant documents at a high-level before retrieving chunks vs. retrieving chunks directly (that might be in irrelevant documents).

Resources:

- [Table Recursive Retrieval](/python/examples/query_engine/pdf_tables/recursive_retriever)
- [Document Summary Index](/python/examples/index_structs/doc_summary/docsummary)

**2. Embed a sentence, which then links to a window around the sentence.**

This allows for finer-grained retrieval of relevant context (embedding giant chunks leads to “lost in the middle” problems), but also ensures enough context for LLM synthesis.

Resources:

- [Metadata Replacement Postprocessor](/python/examples/node_postprocessor/metadatareplacementdemo)

### Structured Retrieval for Larger Document Sets

![](./../_static/production_rag/structured_retrieval.png)

##### Motivation

A big issue with the standard RAG stack (top-k retrieval + basic text splitting) is that it doesn’t do well as the number of documents scales up - e.g. if you have 100 different PDFs.
In this setting, given a query you may want to use structured information to help with more precise retrieval; for instance, if you ask a question that's only relevant to two PDFs,
using structured information to ensure those two PDFs get returned beyond raw embedding similarity with chunks.

##### Key Techniques

There’s a few ways of performing more structured tagging/retrieval for production-quality RAG systems, each with their own pros/cons.

**1. Metadata Filters + Auto Retrieval**
Tag each document with metadata and then store in a vector database. During inference time, use the LLM to infer the right metadata filters to query the vector db in addition to the semantic query string.

- Pros ✅: Supported in major vector dbs. Can filter document via multiple dimensions.
- Cons 🚫: Can be hard to define the right tags. Tags may not contain enough relevant information for more precise retrieval. Also tags represent keyword search at the document-level, doesn’t allow for semantic lookups.

Resources:
**2. Store Document Hierarchies (summaries -> raw chunks) + Recursive Retrieval**
Embed document summaries and map to chunks per document. Fetch at the document-level first before chunk level.

- Pros ✅: Allows for semantic lookups at the document level.
- Cons 🚫: Doesn’t allow for keyword lookups by structured tags (can be more precise than semantic search). Also autogenerating summaries can be expensive.

**Resources**

- [Chroma Auto-Retrieval](/python/examples/vector_stores/chroma_auto_retriever)
- [Document Summary Index](/python/examples/index_structs/doc_summary/docsummary)
- [Auto-Retriever vs. Recursive Retriever](/python/examples/retrievers/auto_vs_recursive_retriever)

### Dynamically Retrieve Chunks Depending on your Task

![](./../_static/production_rag/joint_qa_summary.png)

##### Motivation

RAG isn't just about question-answering about specific facts, which top-k similarity is optimized for. There can be a broad range of queries that a user might ask. Queries that are handled by naive RAG stacks include ones that ask about specific facts e.g. "Tell me about the D&I initiatives for this company in 2023" or "What did the narrator do during his time at Google". But queries can also include summarization e.g. "Can you give me a high-level overview of this document", or comparisons "Can you compare/contrast X and Y". All of these use cases may require different retrieval techniques.

##### Key Techniques

LlamaIndex provides some core abstractions to help you do task-specific retrieval. This includes our [router](/python/framework/module_guides/querying/router) module as well as our [data agent](/python/framework/module_guides/deploying/agents) module.
This also includes some advanced query engine modules.
This also include other modules that join structured and unstructured data.

You can use these modules to do joint question-answering and summarization, or even combine structured queries with unstructured queries.

**Core Module Resources**

- [Query engine](/python/framework/module_guides/deploying/query_engine)
- [Agents](/python/framework/module_guides/deploying/agents)
- [Router](/python/framework/module_guides/querying/router)

**Detailed Guide Resources**

- [Sub-Question Query Engine](/python/examples/query_engine/sub_question_query_engine)
- [Joint QA-Summary](/python/examples/query_engine/jointqasummary)
- [Router Query Engine](/python/examples/query_engine/routerqueryengine)
- [OpenAI Agent Cookbook](/python/examples/agent/openai_agent_query_cookbook)

### Reranking Retrieved Nodes for Precision

##### Motivation

Dense retrievers and hybrid retrievers are tuned for recall; they'll return candidates that are semantically close to the query, but the top-1 isn't always the most relevant. In production, this shows up as "the retrieval looks right, but the model is still answering the wrong question." A reranker addresses this by running a stronger, slower model over the top-`k` candidates and reordering them before they reach the LLM.

##### Key Techniques

Retrieve a wider candidate set (e.g. `similarity_top_k=10-20`) and rerank down to a smaller `top_n` (3–5). Options, in order of cost:

1. Local cross-encoder via [`SentenceTransformerRerank`](/python/framework/module_guides/querying/node_postprocessors/node_postprocessors#sentencetransformerrerank), the recommended default. No API key, runs on the same infra as your embeddings. Use a model like `cross-encoder/ms-marco-MiniLM-L6-v2` for speed or `Qwen/Qwen3-Reranker-0.6B` for stronger multilingual quality.
2. Hosted reranker APIs: `CohereRerank`, `JinaRerank`, `VoyageAIRerank`, `MixedbreadAIRerank`. Simple to wire up, no infra to manage, adds per-query latency and cost.
3. LLM-as-reranker: `LLMRerank`, `RankGPTRerank`, `RankLLMRerank`. Strongest quality, most expensive. Reserve for production flows where accuracy dominates cost.

See the [rerankers overview](/python/framework/module_guides/models/rerankers) for a decision tree and the [node postprocessors guide](/python/framework/module_guides/querying/node_postprocessors/node_postprocessors) for the full list of integrations.

### Optimize Context Embeddings

##### Motivation

This is related to the motivation described above in "decoupling chunks used for retrieval vs. synthesis".
We want to make sure that the embeddings are optimized for better retrieval over your specific data corpus.
Pre-trained models may not capture the salient properties of the data relevant to your use case.

#### Key Techniques

Beyond some of the techniques listed above, we can also try finetuning the embedding model.
We can actually do this over an unstructured text corpus, in a label-free way.

Check out our guides here:

- [Embedding Fine-tuning Guide](/python/examples/finetuning/embeddings/finetune_embedding)

# docs/src/content/docs/framework/optimizing/basic_strategies/basic_strategies.md

---
title: Basic Strategies
---

There are many easy things to try, when you need to quickly squeeze out extra performance and optimize your RAG workflow.

### Prompt Engineering

If you're encountering failures related to the LLM, like hallucinations or poorly formatted outputs, then this
should be one of the first things you try.

Some tasks are listed below, from simple to advanced.

1. Try inspecting the prompts used in your RAG workflow (e.g. the question–answering prompt) and customizing it.

- [Customizing Prompts](/python/examples/prompts/prompt_mixin)
- [Advanced Prompts](/python/examples/prompts/advanced_prompts)

2. Try adding **prompt functions**, allowing you to dynamically inject few-shot examples or process the injected inputs.

- [Advanced Prompts](/python/examples/prompts/advanced_prompts)

### Embeddings

Choosing the right embedding model plays a large role in overall performance.

- Maybe you want a stronger or more recent model than OpenAI's `text-embedding-ada-002` (still the `OpenAIEmbedding` default for backwards compatibility)?
- Maybe you want to run a model locally rather than call a hosted API?
- Maybe you need an embedding model that works well for a specific language?

Beyond OpenAI, many options exist for embedding APIs, running your own embedding model locally, or hosting your own server.

The [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is the best place to compare current models across tons of datasets and tasks. At the time of writing, popular open-source picks include:

- Fast / small baseline (~22-33M params): [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) or [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5). Tiny, fast, still solid for many workloads.
- Mid-size, recent architecture (~150M params): [`lightonai/DenseOn`](https://huggingface.co/lightonai/DenseOn), released April 2026. Sits between the small baselines and the Qwen3 family in size, worth trying if you want recent state-of-the-art retrieval without going to a 0.6B+ model.
- Strong all-around, multilingual (~0.6-4B params): [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (or `-4B` for higher accuracy). Competitive on MTEB, small enough to serve on a single GPU.
- Frictionless hosted API: `OpenAIEmbedding(model="text-embedding-3-small")`.

All of the local/open-source options above can be used via [`HuggingFaceEmbedding`](/python/framework/module_guides/models/embeddings#local-embedding-models) and benefit from ONNX / OpenVINO acceleration on CPU; see the [embeddings guide](/python/framework/module_guides/models/embeddings) for details.

**NOTE:** Unlike an LLM (which you can change at any time), if you change your embedding model, you must re-index your data. Furthermore, you should ensure the same embedding model is used for both indexing and querying.

We have a list of [all supported embedding model integrations](/python/framework/module_guides/models/embeddings).

### Chunk Sizes

Depending on the type of data you are indexing, or the results from your retrieval, you may want to customize the chunk size or chunk overlap.

When documents are ingested into an index, they are split into chunks with a certain amount of overlap. The default chunk size is 1024, while the default chunk overlap is 20.

Changing either of these parameters will change the embeddings that are calculated. A smaller chunk size means the embeddings are more precise, while a larger chunk size means that the embeddings may be more general, but can miss fine-grained details.

We have done our own [initial evaluation on chunk sizes here](https://blog.llamaindex.ai/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5).

Furthermore, when changing the chunk size for a vector index, you may also want to increase the `similarity_top_k` parameter to better represent the amount of data to retrieve for each query.

Here is a full example:

```
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings

documents = SimpleDirectoryReader("./data").load_data()

Settings.chunk_size = 512
Settings.chunk_overlap = 50

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(similarity_top_k=4)
```

Since we halved the default chunk size, the example also doubles the `similarity_top_k` from the default of 2 to 4.

### Hybrid Search

Hybrid search is a common term for retrieval that involves combining results from both semantic search (i.e. embedding similarity) and keyword search.

Embeddings are not perfect, and may fail to return text chunks with matching keywords in the retrieval step.

The solution to this issue is often hybrid search. In LlamaIndex, there are two main ways to achieve this:

1. Use a vector database that has a hybrid search functionality (see [our complete list of supported vector stores](/python/framework/module_guides/storing/vector_stores)).
2. Set up a local hybrid search mechanism with BM25.

Relevant guides with both approaches can be found below:

- [BM25 Retriever](/python/examples/retrievers/bm25_retriever)
- [Reciprocal Rerank Query Fusion](/python/examples/retrievers/reciprocal_rerank_fusion)
- [Weaviate Hybrid Search](/python/examples/vector_stores/weaviateindexdemo-hybrid)
- [Pinecone Hybrid Search](/python/examples/vector_stores/pineconeindexdemo-hybrid)
- [Milvus Hybrid Search](/python/examples/vector_stores/milvushybridindexdemo)

### Reranking

Reranking is one of the highest-leverage knobs for RAG quality. The retriever returns a wider set of candidates (`similarity_top_k=10` or more) and a reranker (a stronger but slower model) re-orders them so the best nodes reach the LLM. It's often the difference between "retrieval looks roughly right" and "retrieval is actually answering the question."

The frictionless recipe uses a local cross-encoder via [`SentenceTransformerRerank`](/python/framework/module_guides/querying/node_postprocessors/node_postprocessors#sentencetransformerrerank); no API key, runs anywhere:

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L6-v2", top_n=3
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)
```

For stronger quality on multilingual content, swap the model for `Qwen/Qwen3-Reranker-0.6B`. For a hosted-API option with minimal setup, use `CohereRerank`, `JinaRerank`, or `VoyageAIRerank`. For the highest quality when latency isn't critical, an LLM-based reranker (`LLMRerank`, `RankGPTRerank`) can outperform cross-encoders at the cost of extra LLM calls.

See the [node postprocessors guide](/python/framework/module_guides/querying/node_postprocessors/node_postprocessors) for the full list and the [rerankers overview](/python/framework/module_guides/models/rerankers) for a decision tree.

### Metadata Filters

Before throwing your documents into a vector index, it can be useful to attach metadata to them. While this metadata can be used later on to help track the sources to answers from the `response` object, it can also be used at query time to filter data before performing the top-k similarity search.

Metadata filters can be set manually, so that only nodes with the matching metadata are returned:

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

documents = [
    Document(text="text", metadata={"author": "LlamaIndex"}),
    Document(text="text", metadata={"author": "John Doe"}),
]

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="author", value="John Doe")]
)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(filters=filters)
```

If you are using an advanced LLM like GPT-4, and your [vector database supports filtering](/python/framework/module_guides/storing/vector_stores), you can get the LLM to write filters automatically at query time, using an `AutoVectorRetriever`.

- [Vector Store Guide](/python/framework/module_guides/indexing/vector_store_guide)

### Document/Node Usage

Take a look at our in-depth guides for more details on how to use Documents/Nodes.

- [Documents Usage](/python/framework/module_guides/loading/documents_and_nodes/usage_documents)
- [Nodes Usage](/python/framework/module_guides/loading/documents_and_nodes/usage_nodes)
- [Metadata Extraction](/python/framework/module_guides/loading/documents_and_nodes/usage_metadata_extractor)

### Multi-Tenancy RAG

Multi-Tenancy in RAG systems is crucial for ensuring data security. It enables users to access exclusively their indexed documents, thereby preventing unauthorized sharing and safeguarding data privacy. Search operations are confined to the user's own data, protecting sensitive information. Implementation can be achieved with `VectorStoreIndex` and `VectorDB` providers through Metadata Filters.

Refer the guides below for more details.

- [Multi Tenancy RAG](/python/examples/multi_tenancy/multi_tenancy_rag)

For detailed guidance on implementing Multi-Tenancy RAG with LlamaIndex and Qdrant, refer to the [blog post](https://qdrant.tech/documentation/tutorials/llama-index-multitenancy/) released by Qdrant.
