# Anthropic cookbook (retrieval notebooks)

# capabilities/retrieval_augmented_generation/guide.ipynb

## Retrieval Augmented Generation

Claude excels at a wide range of tasks, but it may struggle with queries specific to your unique business context. This is where Retrieval Augmented Generation (RAG) becomes invaluable. RAG enables Claude to leverage your internal knowledge bases or customer support documents, significantly enhancing its ability to answer domain-specific questions. Enterprises are increasingly building RAG applications to improve workflows in customer support, Q&A over internal company documents, financial & legal analysis, and much more.

In this guide, we'll demonstrate how to build and optimize a RAG system using the Claude Documentation as our knowledge base. We'll walk you through:

1) Setting up a basic RAG system using an in-memory vector database and embeddings from [Voyage AI](https://www.voyageai.com/).

2) Building a robust evaluation suite. We'll go beyond 'vibes' based evals and show you how to measure the retrieval pipeine & end to end performance independently.

3) Implementing advanced techniques to improve RAG including summary indexing and re-ranking with Claude.

Through a series of targeted improvements, we achieved significant performance gains on the following metrics compared to a basic RAG pipeline (we'll explain what all these metrics *mean* in a bit). These numbers come from the recorded run, which used Claude 3 Haiku for generation and Claude 3.5 Sonnet as the judge, so expect different absolute values when you re-run on current models.

- Avg Precision: 0.43 --> 0.44
- Avg Recall: 0.66 --> 0.69
- Avg F1 Score: 0.52 --> 0.54
- Avg Mean Reciprocal Rank (MRR): 0.74 --> 0.87
- End-to-End Accuracy: 71% --> 81%

##### Note:

The evaluations in this cookbook are meant to mirror a production evaluation system, and you should keep in mind that they can take a while to run. Also of note: if you run the evaluations in full, you may come up against [rate limits](https://docs.claude.com/en/api/rate-limits) depending on your usage tier. Consider skipping the full end to end eval if you're trying to conserve token usage.

### Table of Contents

1) Setup

2) Level 1 - Basic RAG

3) Building an Evaluation System

4) Level 2 - Summary Indexing

5) Level 3 - Summary Indexing and Re-Ranking

### Setup

We'll need a few libraries, including:

1) `anthropic` - to interact with Claude

2) `voyageai` - to generate high quality embeddings

3) `pandas`, `numpy`, `matplotlib`, and `scikit-learn` for data manipulation and visualization


You'll also need API keys from [Anthropic](https://www.anthropic.com/) and [Voyage AI](https://www.voyageai.com/)

```python
## setup
%pip install anthropic
%pip install voyageai
%pip install pandas
%pip install numpy
%pip install matplotlib
%pip install seaborn
%pip install -U scikit-learn
```

```python
import os

os.environ["VOYAGE_API_KEY"] = "VOYAGE KEY HERE"
os.environ["ANTHROPIC_API_KEY"] = "ANTHROPIC KEY HERE"
```

```python
import os

import anthropic

client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def get_text(response):
    """Concatenate the text blocks of a response.

    Current models may return a leading thinking block, so don't index content[0].
    """
    return "".join(block.text for block in response.content if block.type == "text")
```

#### Initialize a Vector DB Class

In this example, we're using an in-memory vector DB, but for a production application, you may want to use a hosted solution. 

```python
import json
import os
import pickle

import numpy as np
import voyageai


class VectorDB:
    def __init__(self, name, api_key=None):
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.client = voyageai.Client(api_key=api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/vector_db.pkl"

    def load_data(self, data):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts = [f"Heading: {item['chunk_heading']}\n\n Chunk Text:{item['text']}" for item in data]
        self._embed_and_store(texts, data)
        self.save_db()
        print("Vector database loaded and saved.")

    def _embed_and_store(self, texts, data):
        batch_size = 128
        result = [
            self.client.embed(texts[i : i + batch_size], model="voyage-2").embeddings
            for i in range(0, len(texts), batch_size)
        ]
        self.embeddings = [embedding for batch in result for embedding in batch]
        self.metadata = data

    def search(self, query, k=5, similarity_threshold=0.75):
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1]
        top_examples = []

        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                example = {
                    "metadata": self.metadata[idx],
                    "similarity": similarities[idx],
                }
                top_examples.append(example)

                if len(top_examples) >= k:
                    break
        self.save_db()
        return top_examples

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError(
                "Vector database file not found. Use load_data to create a new database."
            )
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])
```

### Level 1 - Basic RAG

To get started, we'll set up a basic RAG pipeline using a bare bones approach. This is sometimes called 'Naive RAG' by many in the industry. A basic RAG pipeline includes the following 3 steps:

1) Chunk documents by heading - containing only the content from each subheading

2) Embed each document

3) Use Cosine similarity to retrieve documents in order to answer query

```python
import json
import logging
import re
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the evaluation dataset
with open("evaluation/docs_evaluation_dataset.json") as f:
    eval_data = json.load(f)

# Load the Claude Documentation
with open("data/anthropic_docs.json") as f:
    anthropic_docs = json.load(f)

# Initialize the VectorDB
db = VectorDB("anthropic_docs")
db.load_data(anthropic_docs)


def retrieve_base(query, db):
    results = db.search(query, k=3)
    context = ""
    for result in results:
        chunk = result["metadata"]
        context += f"\n{chunk['text']}\n"
    return results, context


def answer_query_base(query, db):
    documents, context = retrieve_base(query, db)
    prompt = f"""
    You have been tasked with helping us to answer the following query:
    <query>
    {query}
    </query>
    You have access to the following documents which are meant to provide context as you answer the query:
    <documents>
    {context}
    </documents>
    Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already.
    Answer the question now, and avoid providing preamble such as 'Here is the answer', etc
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return get_text(response)
```

### Eval Setup

When evaluating RAG applications, it's critical to evaluate the performance of the retrieval system and end to end system separately.

We synthetically generated an evaluation dataset consisting of 100 samples which include the following:
- A question
- Chunks from our docs which are relevant to that question. This is what we expect our retrieval system to retrieve when the question is asked
- A correct answer to the question.

This is a relatively challenging dataset. Some of our questions require synthesis between more than one chunk in order to be answered correctly, so it's important that our system can load in more than one chunk at a time. You can inspect the dataset by opening `evaluation/docs_evaluation_dataset.json`

Run the next cell to see a preview of the dataset

```python
# previewing our eval dataset
import json


def preview_json(file_path, num_items=3):
    try:
        with open(file_path) as file:
            data = json.load(file)

        if isinstance(data, list):
            preview_data = data[:num_items]
        elif isinstance(data, dict):
            preview_data = dict(list(data.items())[:num_items])
        else:
            print(f"Unexpected data type: {type(data)}. Cannot preview.")
            return

        print(f"Preview of the first {num_items} items from {file_path}:")
        print(json.dumps(preview_data, indent=2))
        print(f"\nTotal number of items: {len(data)}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


preview_json("evaluation/docs_evaluation_dataset.json")
```

## Metric Definitions
We'll evaluate our system based on 5 key metrics: Precision, Recall, F1 Score, Mean Reciprocal Rank (MRR), and End-to-End Accuracy.

### Retrieval Metrics:

#### Precision
Precision represents the proportion of retrieved chunks that are actually relevant. It answers the question: "Of the chunks we retrieved, how many were correct?"

Key points:
- High precision indicates an efficient system with few false positives.
- Low precision suggests many irrelevant chunks are being retrieved.
- Our system retrieves a minimum of 3 chunks per query, which may affect precision scores.

Formula:
$$
\text{Precision} = \frac{\text{True Positives}}{\text{Total Retrieved}} = \frac{|\text{Retrieved} \cap \text{Correct}|}{|\text{Retrieved}|}
$$

#### Recall
Recall measures the completeness of our retrieval system. It answers the question: "Of all the correct chunks that exist, how many did we manage to retrieve?"

Key points:
- High recall indicates comprehensive coverage of necessary information.
- Low recall suggests important chunks are being missed.
- Recall is crucial for ensuring the LLM has access to all needed information.

Formula:
$$
\text{Recall} = \frac{\text{True Positives}}{\text{Total Correct}} = \frac{|\text{Retrieved} \cap \text{Correct}|}{|\text{Correct}|}
$$

#### F1 Score
The F1 score provides a balanced measure between precision and recall. It's particularly useful when you need a single metric to evaluate system performance, especially with uneven class distributions.

Key points:
- F1 score ranges from 0 to 1, with 1 representing perfect precision and recall.
- It's the harmonic mean of precision and recall, tending towards the lower of the two values.
- Useful in scenarios where both false positives and false negatives are important.

Formula:
$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Interpreting F1 score:
- An F1 score of 1.0 indicates perfect precision and recall.
- An F1 score of 0.0 indicates the worst performance.
- Generally, the higher the F1 score, the better the overall performance.

#### Balancing Precision, Recall, and F1 Score:
- There's often a trade-off between precision and recall.
- Our system's minimum chunk retrieval favors recall over precision.
- The optimal balance depends on the specific use case.
- In many RAG systems, high recall is often prioritized, as LLMs can filter out less relevant information during generation.

#### Mean Reciprocal Rank (MRR) @k
MRR measures how well our system ranks relevant information. It helps us understand how quickly a user would find what they're looking for if they started from the top of our retrieved results.

Key points:
- MRR ranges from 0 to 1, where 1 is perfect (correct answer always first).
- It only considers the rank of the first correct result for each query.
- Higher MRR indicates better ranking of relevant information.

Formula:
$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

Where:
- |Q| is the total number of queries
- rank_i is the position of the first relevant item for the i-th query

### End to End Metrics:

#### End to End Accuracy
We use an LLM-as-judge (Claude Sonnet 5.5) to evaluate whether the generated answer is correct based on the question and ground truth answer.

Formula:
$$
\text{End to End Accuracy} = \frac{\text{Number of Correct Answers}}{\text{Total Number of Questions}}
$$

This metric evaluates the entire pipeline, from retrieval to answer generation.

### Defining Our Metric Calculation Functions

```python
def calculate_mrr(retrieved_links: list[str], correct_links: set[str]) -> float:
    for i, link in enumerate(retrieved_links, 1):
        if link in correct_links:
            return 1 / i
    return 0


def evaluate_retrieval(
    retrieval_function: Callable, evaluation_data: list[dict[str, Any]], db: Any
) -> tuple[float, float, float, float, list[float], list[float], list[float]]:
    precisions = []
    recalls = []
    mrrs = []

    for i, item in enumerate(tqdm(evaluation_data, desc="Evaluating Retrieval")):
        try:
            retrieved_chunks, _ = retrieval_function(item["question"], db)
            retrieved_links = [
                chunk["metadata"].get("chunk_link", chunk["metadata"].get("url", ""))
                for chunk in retrieved_chunks
            ]
        except Exception as e:
            logging.error(f"Error in retrieval function: {e}")
            continue

        correct_links = set(item["correct_chunks"])

        true_positives = len(set(retrieved_links) & correct_links)
        precision = true_positives / len(retrieved_links) if retrieved_links else 0
        recall = true_positives / len(correct_links) if correct_links else 0
        mrr = calculate_mrr(retrieved_links, correct_links)

        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)

        if (i + 1) % 10 == 0:
            print(
                f"Processed {i + 1}/{len(evaluation_data)} items. Current Avg Precision: {sum(precisions) / len(precisions):.4f}, Avg Recall: {sum(recalls) / len(recalls):.4f}, Avg MRR: {sum(mrrs) / len(mrrs):.4f}"
            )

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    f1 = (
        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )

    return avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs


def evaluate_end_to_end(answer_query_function, db, eval_data):
    correct_answers = 0
    results = []
    total_questions = len(eval_data)

    for i, item in enumerate(tqdm(eval_data, desc="Evaluating End-to-End")):
        query = item["question"]
        correct_answer = item["correct_answer"]
        generated_answer = answer_query_function(query, db)

        prompt = f"""
        You are an AI assistant tasked with evaluating the correctness of answers to questions about Anthropic's documentation.

        Question: {query}

        Correct Answer: {correct_answer}

        Generated Answer: {generated_answer}

        Is the Generated Answer correct based on the Correct Answer? You should pay attention to the substance of the answer, and ignore minute details that may differ.

        Small differences or changes in wording don't matter. If the generated answer and correct answer are saying essentially the same thing then that generated answer should be marked correct.

        However, if there is any critical piece of information which is missing from the generated answer in comparison to the correct answer, then we should mark this as incorrect.

        Finally, if there are any direct contradictions between the correect answer and generated answer, we should deem the generated answer to be incorrect.

        Respond in the following XML format:
        <evaluation>
        <content>
        <explanation>Your explanation here</explanation>
        <is_correct>true/false</is_correct>
        </content>
        </evaluation>
        """

        try:
            response = client.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
                stop_sequences=["</evaluation>"],
            )

            response_text = get_text(response)
            print(response_text)
            # Pull <is_correct> out directly; the reply may include the opening
            # <evaluation> tag or a short preamble, so don't rely on a fixed root element.
            match = re.search(r"<is_correct>(.*?)</is_correct>", response_text, re.DOTALL)
            if match is None:
                raise ValueError("no <is_correct> tag in judge response")
            is_correct = match.group(1).strip().lower() == "true"

            if is_correct:
                correct_answers += 1
            results.append(is_correct)

            logging.info(f"Question {i + 1}/{total_questions}: {query}")
            logging.info(f"Correct: {is_correct}")
            logging.info("---")

        except ValueError as e:
            logging.error(f"Response parsing error: {e}")
            is_correct = "true" in response_text.lower()
            results.append(is_correct)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            results.append(False)

        if (i + 1) % 10 == 0:
            current_accuracy = correct_answers / (i + 1)
            print(
                f"Processed {i + 1}/{total_questions} questions. Current Accuracy: {current_accuracy:.4f}"
            )
        # time.sleep(2)
    accuracy = correct_answers / total_questions
    return accuracy, results
```

### Helper Function to Plot Performance

```python
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns


def plot_performance(results_folder="evaluation/json_results", include_methods=None, colors=None):
    # Set default colors
    default_colors = ["skyblue", "lightgreen", "salmon"]
    if colors is None:
        colors = default_colors

    # Load JSON files
    results = []
    for filename in os.listdir(results_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(results_folder, filename)
            with open(file_path) as f:
                try:
                    data = json.load(f)
                    if "name" not in data:
                        print(f"Warning: {filename} does not contain a 'name' field. Skipping.")
                        continue
                    if include_methods is None or data["name"] in include_methods:
                        results.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: {filename} is not a valid JSON file. Skipping.")

    if not results:
        print("No JSON files found with matching 'name' fields.")
        return

    # Validate data
    required_metrics = [
        "average_precision",
        "average_recall",
        "average_f1",
        "average_mrr",
        "end_to_end_accuracy",
    ]
    for result in results.copy():
        if not all(metric in result for metric in required_metrics):
            print(f"Warning: {result['name']} is missing some required metrics. Skipping.")
            results.remove(result)

    if not results:
        print("No valid results remaining after validation.")
        return

    # Sort results based on end-to-end accuracy
    results.sort(key=lambda x: x["end_to_end_accuracy"])

    # Prepare data for plotting
    methods = [result["name"] for result in results]
    metrics = required_metrics

    # Set up the plot
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")

    x = range(len(metrics))
    width = 0.8 / len(results)

    # Create color palette
    num_methods = len(methods)
    color_palette = colors[:num_methods] + sns.color_palette("husl", num_methods - len(colors))

    # Plot bars for each method
    for i, (result, color) in enumerate(zip(results, color_palette, strict=False)):
        values = [result[metric] for metric in metrics]
        offset = (i - len(results) / 2 + 0.5) * width
        bars = plt.bar([xi + offset for xi in x], values, width, label=result["name"], color=color)

        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Customize the plot
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Values", fontsize=12)
    plt.title("RAG Performance Metrics (Sorted by End-to-End Accuracy)", fontsize=16)
    plt.xticks(x, metrics, rotation=45, ha="right")
    plt.legend(title="Methods", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()
```

### Evaluating Our Base Case

```python
import pandas as pd

avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs = evaluate_retrieval(
    retrieve_base, eval_data, db
)
e2e_accuracy, e2e_results = evaluate_end_to_end(answer_query_base, db, eval_data)

# Create a DataFrame
df = pd.DataFrame(
    {
        "question": [item["question"] for item in eval_data],
        "retrieval_precision": precisions,
        "retrieval_recall": recalls,
        "retrieval_mrr": mrrs,
        "e2e_correct": e2e_results,
    }
)

# Save to CSV
df.to_csv("evaluation/csvs/evaluation_results_detailed.csv", index=False)
print("Detailed results saved to evaluation/csvs/evaluation_results_one.csv")

# Print the results
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average MRR: {avg_mrr:.4f}")
print(f"Average F1: {f1:.4f}")
print(f"End-to-End Accuracy: {e2e_accuracy:.4f}")

# Save the results to a file
with open("evaluation/json_results/evaluation_results_one.json", "w") as f:
    json.dump(
        {
            "name": "Basic RAG",
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": f1,
            "average_mrr": avg_mrr,
            "end_to_end_accuracy": e2e_accuracy,
        },
        f,
        indent=2,
    )

print(
    "Evaluation complete. Results saved to evaluation_results_one.json, evaluation_results_one.csv"
)
```

```python
# let's visualize our performance
plot_performance("evaluation/json_results", ["Basic RAG"], colors=["skyblue"])
```

## Level 2: Document Summarization for Enhanced Retrieval

In this section, we'll implement an improved approach to our retrieval system by incorporating document summaries. Instead of embedding chunks directly from the documents, we'll create a concise summary for each chunk and use this summary along with the original content in our embedding process.

This approach aims to capture the essence of each document chunk more effectively, potentially leading to improved retrieval performance.

Key steps in this process:
1. We load the original document chunks.
2. For each chunk, we generate a 2-3 sentence summary using Claude.
3. We store both the original content and the summary for each chunk in a new json file: `data/anthropic_summary_indexed_docs.json`

This summary-enhanced approach is designed to provide more context during the embedding and retrieval phases, potentially improving the system's ability to understand and match the most relevant documents to user queries.

### Generating the Summaries and Storing Them

```python
import json

from tqdm import tqdm


def generate_summaries(input_file, output_file):
    # Load the original documents
    with open(input_file) as f:
        docs = json.load(f)

    # Prepare the context about the overall knowledge base
    knowledge_base_context = "This is documentation for Anthropic's, a frontier AI lab building Claude, an LLM that excels at a variety of general purpose tasks. These docs contain model details and documentation on Anthropic's APIs."

    summarized_docs = []

    for doc in tqdm(docs, desc="Generating summaries"):
        prompt = f"""
        You are tasked with creating a short summary of the following content from Anthropic's documentation.

        Context about the knowledge base:
        {knowledge_base_context}

        Content to summarize:
        Heading: {doc["chunk_heading"]}
        {doc["text"]}

        Please provide a brief summary of the above content in 2-3 sentences. The summary should capture the key points and be concise. We will be using it as a key part of our search pipeline when answering user queries about this content.

        Avoid using any preamble whatsoever in your response. Statements such as 'here is the summary' or 'the summary is as follows' are prohibited. You should get straight into the summary itself and be concise. Every word matters.
        """

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        summary = get_text(response).strip()

        summarized_doc = {
            "chunk_link": doc["chunk_link"],
            "chunk_heading": doc["chunk_heading"],
            "text": doc["text"],
            "summary": summary,
        }
        summarized_docs.append(summarized_doc)

    # Save the summarized documents to a new JSON file
    with open(output_file, "w") as f:
        json.dump(summarized_docs, f, indent=2)

    print(f"Summaries generated and saved to {output_file}")


# generate_summaries('data/anthropic_docs.json', 'data/anthropic_summary_indexed_docs.json')
```

## Summary-Indexed Vector Database Creation

Here, we're creating a new vector database that incorporates our summary-enhanced document chunks. This approach combines the original text, the chunk heading, and the newly generated summary into a single text for embedding.

Key features of this process:
1. We create embeddings for the combined text (heading + summary + original content) using the Voyage AI API.
2. The embeddings and full metadata (including summaries) are stored in our vector database.
3. We implement caching mechanisms to improve efficiency in repeated queries.
4. The database is saved to disk for persistence and quick loading in future sessions.

This summary-indexed approach aims to create more informative embeddings, potentially leading to more accurate and contextually relevant document retrieval.

```python
import json
import os
import pickle

import numpy as np
import voyageai


class SummaryIndexedVectorDB:
    def __init__(self, name, api_key=None):
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.client = voyageai.Client(api_key=api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/summary_indexed_vector_db.pkl"

    def load_data(self, data_file):
        # Check if the vector database is already loaded
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        # Check if vector_db.pkl exists
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        with open(data_file) as f:
            data = json.load(f)

        texts = [
            f"{item['chunk_heading']}\n\n{item['text']}\n\n{item['summary']}" for item in data
        ]  # Embed Chunk Heading + Text + Summary Together
        # Embed more than 128 documents with a for loop
        batch_size = 128
        result = [
            self.client.embed(texts[i : i + batch_size], model="voyage-2").embeddings
            for i in range(0, len(texts), batch_size)
        ]

        # Flatten the embeddings
        self.embeddings = [embedding for batch in result for embedding in batch]
        self.metadata = data  # Store the entire item as metadata
        self.save_db()
        # Save the vector database to disk
        print("Vector database loaded and saved.")

    def search(self, query, k=3, similarity_threshold=0.75):
        query_embedding = None
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1]
        top_examples = []

        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                example = {
                    "metadata": self.metadata[idx],
                    "similarity": similarities[idx],
                }
                top_examples.append(example)

                if len(top_examples) >= k:
                    break
        self.save_db()
        return top_examples

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError(
                "Vector database file not found. Use load_data to create a new database."
            )

        with open(self.db_path, "rb") as file:
            data = pickle.load(file)

        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])
```

## Enhanced Retrieval Using Summary-Indexed Embeddings

In this section, we implement the retrieval process using our new summary-indexed vector database. This approach leverages the enhanced embeddings we created, which incorporate document summaries along with the original content.

Key aspects of this updated retrieval process:
1. We search the vector database using the query embedding, retrieving the top k most similar documents.
2. For each retrieved document, we include the chunk heading, summary, and full text in the context provided to the LLM.
3. This enriched context is then used to generate an answer to the user's query.

By including summaries in both the embedding and retrieval phases, we aim to provide the LLM with a more comprehensive and focused context. This could potentially lead to more accurate and relevant answers, as the LLM has access to both a concise overview (the summary) and the detailed information (the full text) for each relevant document chunk.

```python
def retrieve_level_two(query, db):
    results = db.search(query, k=3)
    context = ""
    for result in results:
        chunk = result["metadata"]
        context += f"\n <document> \n {chunk['chunk_heading']}\n\nText\n {chunk['text']} \n\nSummary: \n {chunk['summary']} \n </document> \n"  # show model all 3 items
    return results, context


def answer_query_level_two(query, db):
    documents, context = retrieve_base(query, db)
    prompt = f"""
    You have been tasked with helping us to answer the following query:
    <query>
    {query}
    </query>
    You have access to the following documents which are meant to provide context as you answer the query:
    <documents>
    {context}
    </documents>
    Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already.
    Answer the question now, and avoid providing preamble such as 'Here is the answer', etc
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return get_text(response)
```

```python
# Initialize the SummaryIndexedVectorDB
level_two_db = SummaryIndexedVectorDB("anthropic_docs_v2")
level_two_db.load_data("data/anthropic_summary_indexed_docs.json")

import pandas as pd

# Run the evaluations
avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs = evaluate_retrieval(
    retrieve_level_two, eval_data, level_two_db
)
e2e_accuracy, e2e_results = evaluate_end_to_end(answer_query_level_two, level_two_db, eval_data)

# Create a DataFrame
df = pd.DataFrame(
    {
        "question": [item["question"] for item in eval_data],
        "retrieval_precision": precisions,
        "retrieval_recall": recalls,
        "retrieval_mrr": mrrs,
        "e2e_correct": e2e_results,
    }
)

# Save to CSV
df.to_csv("evaluation/csvs/evaluation_results_detailed_level_two.csv", index=False)
print("Detailed results saved to evaluation_results_detailed.csv")

# Print the results
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average MRR: {avg_mrr:.4f}")
print(f"Average F1: {f1:.4f}")
print(f"End-to-End Accuracy: {e2e_accuracy:.4f}")

# Save the results to a file
with open("evaluation/json_results/evaluation_results_level_two.json", "w") as f:
    json.dump(
        {
            "name": "Summary Indexing",
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": f1,
            "average_mrr": avg_mrr,
            "end_to_end_accuracy": e2e_accuracy,
        },
        f,
        indent=2,
    )

print(
    "Evaluation complete. Results saved to evaluation_results_level_two.json, evaluation_results_detailed_level_two.csv"
)
```

### Evaluating This Method vs Basic RAG

```python
# visualizing our performance
plot_performance("evaluation/json_results", ["Basic RAG", "Summary Indexing"])
```

### Level 3 - Re-Ranking with Claude
In this final enhancement to our retrieval system, we introduce a reranking step to further improve the relevance of the retrieved documents. This approach leverages Claude's power to better understand the context and nuances of both the query and the retrieved documents.

The `rerank_results` function uses Claude to reassess and reorder the initially retrieved documents:
1. It presents Claude with the query and summaries of all retrieved documents.
2. Claude is asked to select and rank the most relevant documents.
3. The function parses Claude's response to get the reranked document indices.
4. It includes fallback mechanisms in case of errors or insufficient results.
5. Finally, it assigns descending relevance scores to the reranked results.

The `retrieve_advanced` function implements the new retrieval pipeline:
1. We initially retrieve more documents than needed (default 20, configurable via `initial_k`) from the vector database.
2. We then use the `rerank_results` function to refine this larger set down to the most relevant documents (default 3, configurable via `k`).
3. Finally, it generates a new context string from these reranked documents.

This process casts a wider net initially and then uses AI to focus on the most pertinent information. By combining vector-based retrieval with LLM reranking, this approach aims to provide more accurate and contextually appropriate responses to user queries.

In the recorded run (Claude 3 Haiku generating, Claude 3.5 Sonnet judging), our evaluations showed significant improvements:
- Accuracy increased from 78% in our previous system to 85%.
- Precision was improved by using our re-ranker to reduce the number of documents shown to the LLM.
- MRR (Mean Reciprocal Rank) was likely improved by asking Claude to rank the relevance of each document in order.

These improvements demonstrate the effectiveness of incorporating AI-powered reranking in our retrieval process.

```python
def rerank_results(query: str, results: list[dict], k: int = 5) -> list[dict]:
    # Prepare the summaries with their indices
    summaries = []
    print(len(results))

    for i, result in enumerate(results):
        summary = f"[{i}] Document Summary: {result['metadata']['summary']}"
        summaries.append(summary)
    joined_summaries = "\n\n".join(summaries)

    prompt = f"""
    Query: {query}
    You are about to be given a group of documents, each preceded by its index number in square brackets. Your task is to select the only {k} most relevant documents from the list to help us answer the query.

    <documents>
    {joined_summaries}
    </documents>

    Output only the indices of {k} most relevant documents in order of relevance, separated by commas, enclosed in XML tags here:
    <relevant_indices>put the numbers of your indices here, seeparted by commas</relevant_indices>
    """
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=50,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "<relevant_indices>"},
            ],
            temperature=0,
            stop_sequences=["</relevant_indices>"],
        )

        # Extract the indices from the response
        response_text = get_text(response).strip()
        indices_str = response_text
        relevant_indices = []
        for idx in indices_str.split(","):
            try:
                relevant_indices.append(int(idx.strip()))
            except ValueError:
                continue  # Skip invalid indices
        print(indices_str)
        print(relevant_indices)
        # If we didn't get enough valid indices, fall back to the top k by original order
        if len(relevant_indices) == 0:
            relevant_indices = list(range(min(k, len(results))))

        # Ensure we don't have out-of-range indices
        relevant_indices = [idx for idx in relevant_indices if idx < len(results)]

        # Return the reranked results
        reranked_results = [results[idx] for idx in relevant_indices[:k]]
        # Assign descending relevance scores
        for i, result in enumerate(reranked_results):
            result["relevance_score"] = (
                100 - i
            )  # Highest score is 100, decreasing by 1 for each rank

        return reranked_results

    except Exception as e:
        print(f"An error occurred during reranking: {str(e)}")
        # Fall back to returning the top k results without reranking
        return results[:k]


def retrieve_advanced(
    query: str, db: SummaryIndexedVectorDB, k: int = 3, initial_k: int = 20
) -> tuple[list[dict], str]:
    # Step 1: Get initial results
    initial_results = db.search(query, k=initial_k)

    # Step 2: Re-rank results
    reranked_results = rerank_results(query, initial_results, k=k)

    # Step 3: Generate new context string from re-ranked results
    new_context = ""
    for result in reranked_results:
        chunk = result["metadata"]
        new_context += (
            f"\n <document> \n {chunk['chunk_heading']}\n\n{chunk['text']} \n </document> \n"
        )

    return reranked_results, new_context


# The answer_query_advanced function remains unchanged
def answer_query_advanced(query: str, db: SummaryIndexedVectorDB):
    documents, context = retrieve_advanced(query, db)
    prompt = f"""
    You have been tasked with helping us to answer the following query:
    <query>
    {query}
    </query>
    You have access to the following documents which are meant to provide context as you answer the query:
    <documents>
    {context}
    </documents>
    Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already.
    Answer the question now, and avoid providing preamble such as 'Here is the answer', etc
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return get_text(response)
```

### Evaluation

```python
# Initialize the SummaryIndexedVectorDB
level_three_db = SummaryIndexedVectorDB("anthropic_docs_v3")
level_three_db.load_data("data/anthropic_summary_indexed_docs.json")

import pandas as pd

# Run the evaluations
avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs = evaluate_retrieval(
    retrieve_advanced, eval_data, level_three_db
)
e2e_accuracy, e2e_results = evaluate_end_to_end(answer_query_advanced, level_two_db, eval_data)

# Create a DataFrame
df = pd.DataFrame(
    {
        "question": [item["question"] for item in eval_data],
        "retrieval_precision": precisions,
        "retrieval_recall": recalls,
        "retrieval_mrr": mrrs,
        "e2e_correct": e2e_results,
    }
)

# Save to CSV
df.to_csv("evaluation/csvs/evaluation_results_detailed_level_three.csv", index=False)
print("Detailed results saved to evaluation_results_detailed_level_three.csv")

# Plot the results
# Print the results
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1: {f1:.4f}")
print(f"Average Mean Reciprocal Rank: {avg_mrr:4f}")
print(f"End-to-End Accuracy: {e2e_accuracy:.4f}")

# Save the results to a file
with open("evaluation/json_results/evaluation_results_level_three.json", "w") as f:
    json.dump(
        {
            "name": "Summary Indexing + Re-Ranking",
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": f1,
            "average_mrr": avg_mrr,
            "end_to_end_accuracy": e2e_accuracy,
        },
        f,
        indent=2,
    )

print(
    "Evaluation complete. Results saved to evaluation_results_level_three.json, evaluation_results_detailed_level_three.csv, and evaluation_results_level_three.png"
)
```

```python
# visualizing our performance
plot_performance(
    "evaluation/json_results",
    ["Basic RAG", "Summary Indexing", "Summary Indexing + Re-Ranking"],
    colors=["skyblue", "lightgreen", "salmon"],
)
```

### Evaluation - Going Deeper with Promptfoo

This guide has illustrated the importance of measuring prompt performance empirically when prompt engineering. You can read more about our empirical methodology to prompt engineering here. Using a Jupyter Notebook is a great way to start prompt engineering but as your datasets grow larger and your prompts more numerous it is important to leverage tooling that will scale with you.

In this section of the guide we will explore using Promptfoo an open source LLM evaluation toolkit. To get started head over to the ./evaluation directory and checkout the ./evaluation/README.md.

Promptfoo makes it very easy to build automated test suites that compare different models, hyperparameter choices, and prompts against one another. 

As an example, you can run the below cell to see the average performance of each provider across all of our test cases. The checked-in `data/end_to_end_results.json` is a recorded run that compared Claude 3 Haiku ("Haiku: T-0.0") with Claude 3.5 Sonnet ("3.5 Sonnet: T-0.0"). Re-running the promptfoo eval overwrites it with results for the models in `promptfooconfig_end_to_end.yaml`.

```python
import json

import numpy as np
import pandas as pd

# Load the JSON file
with open("data/end_to_end_results.json") as f:
    data = json.load(f)

# Extract the results
results = data["results"]["results"]

# Create a DataFrame
df = pd.DataFrame(results)

# Extract provider, prompt, and score information
df["provider"] = df["provider"].apply(lambda x: x["label"] if isinstance(x, dict) else x)
df["prompt"] = df["prompt"].apply(lambda x: x["label"] if isinstance(x, dict) else x)


# Function to safely extract scores
def extract_score(x):
    if isinstance(x, dict) and "score" in x:
        return x["score"] * 100  # Convert to percentage
    return np.nan


df["score"] = df["gradingResult"].apply(extract_score)

# Group by provider and prompt, then calculate mean scores
result = df.groupby(["provider", "prompt"])["score"].mean().unstack()

# Fill NaN values with 0
result = result.fillna(0)

# Calculate the average score across all prompts for each provider
result["Average"] = result.mean(axis=1)

# Sort the result by the average score
result = result.sort_values(by="Average", ascending=False)

# Round the results to 2 decimal places
result = result.round(2)
# Calculate overall statistics
overall_average = result["Average"].mean()
overall_std = result["Average"].std()
best_provider = result["Average"].idxmax()
worst_provider = result["Average"].idxmin()

print("\nOverall Statistics:")
print(f"Best Performing Provider: {best_provider} ({result.loc[best_provider, 'Average']:.2f}%)")
print(f"Worst Performing Provider: {worst_provider} ({result.loc[worst_provider, 'Average']:.2f}%)")
```

# capabilities/contextual-embeddings/guide.ipynb

## Enhancing RAG with Contextual Retrieval

> Note: For more background information on Contextual Retrieval, including additional performance evaluations on various datasets, we recommend reading our accompanying  [blog post](https://www.anthropic.com/news/contextual-retrieval).

Retrieval Augmented Generation (RAG) enables Claude to leverage your internal knowledge bases, codebases, or any other corpus of documents when providing a response. Enterprises are increasingly building RAG applications to improve workflows in customer support, Q&A over internal company documents, financial & legal analysis, code generation, and much more.

In a [separate guide](https://github.com/anthropics/anthropic-cookbook/blob/main/capabilities/retrieval_augmented_generation/guide.ipynb), we walked through setting up a basic retrieval system, demonstrated how to evaluate its performance, and then outlined a few techniques to improve performance. In this guide, we present a technique for improving retrieval performance: Contextual Embeddings.

In traditional RAG, documents are typically split into smaller chunks for efficient retrieval. While this approach works well for many applications, it can lead to problems when individual chunks lack sufficient context. Contextual Embeddings solve this problem by adding relevant context to each chunk before embedding. This method improves the quality of each embedded chunk, allowing for more accurate retrieval and thus better overall performance. Averaged across all data sources we tested, Contextual Embeddings reduced the top-20-chunk retrieval failure rate by 35%.

The same chunk-specific context can also be used with BM25 search to further improve retrieval performance. We introduce this technique in the "Contextual BM25" section.

In this guide, we'll demonstrate how to build and optimize a Contextual Retrieval system using a dataset of 9 codebases as our knowledge base. We'll walk through:

1) Setting up a basic retrieval pipeline to establish a baseline for performance.

2) Contextual Embeddings: what it is, why it works, and how prompt caching makes it practical for production use cases.

3) Implementing Contextual Embeddings and demonstrating performance improvements.

4) Contextual BM25: improving performance with *contextual* BM25 hybrid search.

5) Improving performance with reranking,

#### Evaluation Metrics & Dataset:

We use a pre-chunked dataset of 9 codebases - all of which have been chunked according to a basic character splitting mechanism. Our evaluation dataset contains 248 queries - each of which contains a 'golden chunk.' We'll use a metric called Pass@k to evaluate performance. Pass@k checks whether or not the 'golden document' was present in the first k documents retrieved for each query. Contextual Embeddings in this case helped us to improve Pass@10 performance from ~87% --> ~95%.

You can find the code files and their chunks in `data/codebase_chunks.json` and the evaluation dataset in `data/evaluation_set.jsonl`

##### Additional Notes:

Prompt caching is helpful in managing costs when using this retrieval method. This feature is currently available on Anthropic's first-party API, and is coming soon to our third-party partner environments in AWS Bedrock and GCP Vertex. We know that many of our customers leverage AWS Knowledge Bases and GCP Vertex AI APIs when building RAG solutions, and this method can be used on either platform with a bit of customization. Consider reaching out to Anthropic or your AWS/GCP account team for guidance on this!

To make it easier to use this method on Bedrock, the AWS team has provided us with code that you can use to implement a Lambda function that adds context to each document. If you deploy this Lambda function, you can select it as a custom chunking option when configuring a [Bedrock Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html). You can find this code in `contextual-rag-lambda-function`. The main lambda function code is in `lambda_function.py`.

### Table of Contents

1) Setup

2) Basic RAG

3) Contextual Embeddings

4) Contextual BM25

5) Reranking

### Setup

Before starting this guide, ensure you have:

**Technical Skills:**
- Intermediate Python programming
- Basic understanding of RAG (Retrieval Augmented Generation)
- Familiarity with vector databases and embeddings
- Basic command-line proficiency

**System Requirements:**
- Python 3.8+
- Docker installed and running (optional, for BM25 search)
- 4GB+ available RAM
- ~5-10 GB disk space for vector databases

**API Access:**
- [Anthropic API key](https://console.anthropic.com/) (free tier sufficient)
- [Voyage AI API key](https://www.voyageai.com/)
- [Cohere API key](https://cohere.com/)

**Time & Cost:**
- Expected completion time: 30-45 minutes
- API costs: ~$5-10 to run through the full dataset

#### Libraries 

We'll need a few libraries, including:

1) `anthropic` - to interact with Claude

2) `voyageai` - to generate high quality embeddings

3) `cohere` - for reranking

4) `elasticsearch` for performant BM25 search

3) `pandas`, `numpy`, `matplotlib`, and `scikit-learn` for data manipulation and visualization

#### Environment Variables 

Ensure the following environment variables are set:

```
- VOYAGE_API_KEY
- ANTHROPIC_API_KEY
- COHERE_API_KEY
```

```python
%%capture
%pip install --upgrade anthropic voyageai cohere elasticsearch pandas numpy
```

We define our model names up front to make it easier to change models as new models are released

```python
MODEL_NAME = "claude-haiku-4-5"
```

We'll start by initializing the Anthropic client that we'll use for generating contextual descriptions.

```python
import os

import anthropic

client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
```

### Initialize a Vector DB Class

We'll create a VectorDB class to handle embedding storage and similarity search. This class serves three key functions in our RAG pipeline:

1. **Embedding Generation**: Converts text chunks into vector representations using Voyage AI's embedding model
2. **Storage & Caching**: Saves embeddings to disk to avoid re-computing them (which saves time and API costs)
3. **Similarity Search**: Retrieves the most relevant chunks for a given query using cosine similarity


For this guide, we're using a simple in-memory vector database with pickle serialization. This makes the code easy to understand and requires no external dependencies. The class automatically saves embeddings to disk after generation, so you only pay the embedding cost once. 

For production use, consider hosted vector database solutions.

The VectorDB class below follows the same interface patterns you'd use with production solutions, making it easy to swap out later. Key features include batch processing (128 chunks at a time), progress tracking with tqdm, and query caching to speed up repeated searches during evaluation.

```python
import json
import pickle
from typing import Any

import numpy as np
import voyageai
from tqdm import tqdm


class VectorDB:
    def __init__(self, name: str, api_key=None):
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.client = voyageai.Client(api_key=api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/vector_db.pkl"

    def load_data(self, dataset: list[dict[str, Any]]):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc["chunks"]) for doc in dataset)

        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for doc in dataset:
                for chunk in doc["chunks"]:
                    texts_to_embed.append(chunk["content"])
                    metadata.append(
                        {
                            "doc_id": doc["doc_id"],
                            "original_uuid": doc["original_uuid"],
                            "chunk_id": chunk["chunk_id"],
                            "original_index": chunk["original_index"],
                            "content": chunk["content"],
                        }
                    )
                    pbar.update(1)

        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()

        print(f"Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}")

    def _embed_and_store(self, texts: list[str], data: list[dict[str, Any]]):
        batch_size = 128
        with tqdm(total=len(texts), desc="Embedding chunks") as pbar:
            result = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_result = self.client.embed(batch, model="voyage-2").embeddings
                result.extend(batch_result)
                pbar.update(len(batch))

        self.embeddings = result
        self.metadata = data

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]

        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)

        return top_results

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError(
                "Vector database file not found. Use load_data to create a new database."
            )
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])
```

Now we can use this class to load our dataset

```python
# Load your transformed dataset
with open("data/codebase_chunks.json") as f:
    transformed_dataset = json.load(f)

# Initialize the VectorDB
base_db = VectorDB("base_db")

# Load and process the data
base_db.load_data(transformed_dataset)
```

### Basic RAG

To get started, we'll set up a basic RAG pipeline using a bare bones approach. This is sometimes called 'Naive RAG' by many in the industry. A basic RAG pipeline includes the following 3 steps:

1) Chunk documents by heading - containing only the content from each subheading

2) Embed each document

3) Use Cosine similarity to retrieve documents in order to answer query

```python
import json
from collections.abc import Callable
from typing import Any

from tqdm import tqdm


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load JSONL file and return a list of dictionaries."""
    with open(file_path) as file:
        return [json.loads(line) for line in file]


def evaluate_retrieval(
    queries: list[dict[str, Any]], retrieval_function: Callable, db, k: int = 20
) -> dict[str, float]:
    total_score = 0
    total_queries = len(queries)

    for query_item in tqdm(queries, desc="Evaluating retrieval"):
        query = query_item["query"]
        golden_chunk_uuids = query_item["golden_chunk_uuids"]

        # Find all golden chunk contents
        golden_contents = []
        for doc_uuid, chunk_index in golden_chunk_uuids:
            golden_doc = next(
                (doc for doc in query_item["golden_documents"] if doc["uuid"] == doc_uuid), None
            )
            if not golden_doc:
                print(f"Warning: Golden document not found for UUID {doc_uuid}")
                continue

            golden_chunk = next(
                (chunk for chunk in golden_doc["chunks"] if chunk["index"] == chunk_index), None
            )
            if not golden_chunk:
                print(
                    f"Warning: Golden chunk not found for index {chunk_index} in document {doc_uuid}"
                )
                continue

            golden_contents.append(golden_chunk["content"].strip())

        if not golden_contents:
            print(f"Warning: No golden contents found for query: {query}")
            continue

        retrieved_docs = retrieval_function(query, db, k=k)

        # Count how many golden chunks are in the top k retrieved documents
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[:k]:
                retrieved_content = (
                    doc["metadata"]
                    .get("original_content", doc["metadata"].get("content", ""))
                    .strip()
                )
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break

        query_score = chunks_found / len(golden_contents)
        total_score += query_score

    average_score = total_score / total_queries
    pass_at_n = average_score * 100
    return {"pass_at_n": pass_at_n, "average_score": average_score, "total_queries": total_queries}


def retrieve_base(query: str, db, k: int = 20) -> list[dict[str, Any]]:
    """
    Retrieve relevant documents using either VectorDB or ContextualVectorDB.

    :param query: The query string
    :param db: The VectorDB or ContextualVectorDB instance
    :param k: Number of top results to retrieve
    :return: List of retrieved documents
    """
    return db.search(query, k=k)


def evaluate_db(db, original_jsonl_path: str, k):
    # Load the original JSONL data for queries and ground truth
    original_data = load_jsonl(original_jsonl_path)

    # Evaluate retrieval
    results = evaluate_retrieval(original_data, retrieve_base, db, k)
    return results


def evaluate_and_display(db, jsonl_path: str, k_values: list[int] = None, db_name: str = ""):
    """
    Evaluate retrieval performance across multiple k values and display formatted results.

    Args:
        db: Vector database instance (VectorDB or ContextualVectorDB)
        jsonl_path: Path to evaluation dataset
        k_values: List of k values to evaluate (default: [5, 10, 20])
        db_name: Optional name for the database being evaluated

    Returns:
        Dict mapping k values to their results
    """
    if k_values is None:
        k_values = [5, 10, 20]
    results = {}

    print(f"{'=' * 60}")
    if db_name:
        print(f"Evaluation Results: {db_name}")
    else:
        print("Evaluation Results")
    print(f"{'=' * 60}\n")

    for k in k_values:
        print(f"Evaluating Pass@{k}...")
        results[k] = evaluate_db(db, jsonl_path, k)
        print()  # Add spacing between evaluations

    # Print summary table
    print(f"{'=' * 60}")
    print(f"{'Metric':<15} {'Pass Rate':<15} {'Score':<15}")
    print(f"{'-' * 60}")
    for k in k_values:
        pass_rate = f"{results[k]['pass_at_n']:.2f}%"
        score = f"{results[k]['average_score']:.4f}"
        print(f"{'Pass@' + str(k):<15} {pass_rate:<15} {score:<15}")
    print(f"{'=' * 60}\n")

    return results
```

Now let's establish our baseline performance by evaluating the basic RAG system. We'll test at k=5, 10, and 20 to see how many of the golden chunks appear in the top retrieved results. This gives us a benchmark to measure improvement against.


```python
results = evaluate_and_display(
    base_db, "data/evaluation_set.jsonl", k_values=[5, 10, 20], db_name="Baseline RAG"
)
```

These results show our baseline RAG performance. The system successfully retrieves the correct chunk 81% of the time in the top 5 results, improving to 87% in the top 10, and 90% in the top 20.

### Contextual Embeddings

With basic RAG, individual chunks often lack sufficient context when embedded in isolation. Contextual Embeddings solve this by using Claude to generate a brief description that "situates" each chunk within its source document. We then embed the chunk together with this context, creating richer vector representations. 

For each chunk in our codebase dataset, we pass both the chunk and its full source file to Claude. Claude generates a concise explanation of what the chunk contains and where it fits in the overall file. This context gets prepended to the chunk before embedding.


#### Cost and Latency Considerations

**When does this cost occur?** The contextualization happens once at ingestion time, not during every query. Unlike techniques like HyDE (hypothetical document embeddings) that add latency to each search, contextual embeddings are a one-time cost when building your vector database. Prompt caching makes this practical. Since we process all chunks from the same document sequentially, we can leverage prompt caching for significant savings.

1. First chunk: We write the full document to cache (pay a small premium)
2. Subsequent chunks: Read the document from cache (90% discount on those tokens)
3. Cache lasts 5 minutes, plenty of time to process all chunks in a document

**Cost example**: For 800-token chunks in 8k-token documents with 100 tokens of generated context, the total cost is $1.02 per million document tokens. You'll see the cache savings in the logs when you run the code below.

**Note:** Some embedding models have fixed input token limits. If you see worse performance with contextual embeddings, your contextualized chunks may be getting truncated—consider using an embedding model with a larger context window.

--- 

Let's see an example of how contextual embeddings work by generating context for a single chunk. We'll use Claude to create a situating context, and you'll also see the prompt caching metrics in action.


```python
DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""


def situate_context(doc: str, chunk: str) -> str:
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                        "cache_control": {
                            "type": "ephemeral"
                        },  # we will make use of prompt caching for the full documents
                    },
                    {
                        "type": "text",
                        "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                    },
                ],
            }
        ],
    )
    return response


jsonl_data = load_jsonl("data/evaluation_set.jsonl")
# Example usage
doc_content = jsonl_data[0]["golden_documents"][0]["content"]
chunk_content = jsonl_data[0]["golden_chunks"][0]["content"]

response = situate_context(doc_content, chunk_content)
print(f"Situated context: {response.content[0].text}")
print("-" * 10)
# Print cache performance metrics
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Cache creation input tokens: {response.usage.cache_creation_input_tokens}")
print(f"Cache read input tokens: {response.usage.cache_read_input_tokens}")
```

#### Building the Contextual Vector Database

Now that we've seen how to generate contextual descriptions for individual chunks, let's scale this up to process our entire dataset. The `ContextualVectorDB` class below extends our basic `VectorDB` with automatic contextualization during ingestion.

**Key features:**

- **Parallel processing**: Uses ThreadPoolExecutor to contextualize multiple chunks simultaneously (configurable thread count)
- **Automatic prompt caching**: Processes chunks document-by-document to maximize cache hits
- **Token tracking**: Monitors cache performance and calculates actual cost savings
- **Persistent storage**: Saves both embeddings and contextualized metadata to disk

When you run this, pay attention to the token usage statistics—you'll see that 70-80% of input tokens are read from cache, demonstrating the dramatic cost savings from prompt caching. On our 737-chunk dataset, this reduces what would be a ~$15 ingestion job down to ~$3.

```python
import json
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import anthropic
import numpy as np
import voyageai
from tqdm import tqdm


class ContextualVectorDB:
    def __init__(self, name: str, voyage_api_key=None, anthropic_api_key=None):
        if voyage_api_key is None:
            voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        self.voyage_client = voyageai.Client(api_key=voyage_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/contextual_vector_db.pkl"

        self.token_counts = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
        self.token_lock = threading.Lock()

    def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        response = self.anthropic_client.messages.create(
            model=MODEL_NAME,
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                            "cache_control": {
                                "type": "ephemeral"
                            },  # we will make use of prompt caching for the full documents
                        },
                        {
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                        },
                    ],
                },
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        return response.content[0].text, response.usage

    def load_data(self, dataset: list[dict[str, Any]], parallel_threads: int = 1):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc["chunks"]) for doc in dataset)

        def process_chunk(doc, chunk):
            # for each chunk, produce the context
            contextualized_text, usage = self.situate_context(doc["content"], chunk["content"])
            with self.token_lock:
                self.token_counts["input"] += usage.input_tokens
                self.token_counts["output"] += usage.output_tokens
                self.token_counts["cache_read"] += usage.cache_read_input_tokens
                self.token_counts["cache_creation"] += usage.cache_creation_input_tokens

            return {
                # append the context to the original text chunk
                "text_to_embed": f"{contextualized_text}\n\n{chunk['content']}",
                "metadata": {
                    "doc_id": doc["doc_id"],
                    "original_uuid": doc["original_uuid"],
                    "chunk_id": chunk["chunk_id"],
                    "original_index": chunk["original_index"],
                    "original_content": chunk["content"],
                    "contextualized_content": contextualized_text,
                },
            }

        print(f"Processing {total_chunks} chunks with {parallel_threads} threads")
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for doc in dataset:
                for chunk in doc["chunks"]:
                    futures.append(executor.submit(process_chunk, doc, chunk))

            for future in tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks"):
                result = future.result()
                texts_to_embed.append(result["text_to_embed"])
                metadata.append(result["metadata"])

        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()

        # logging token usage
        print(
            f"Contextual Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}"
        )
        print(f"Total input tokens without caching: {self.token_counts['input']}")
        print(f"Total output tokens: {self.token_counts['output']}")
        print(f"Total input tokens written to cache: {self.token_counts['cache_creation']}")
        print(f"Total input tokens read from cache: {self.token_counts['cache_read']}")

        total_tokens = (
            self.token_counts["input"]
            + self.token_counts["cache_read"]
            + self.token_counts["cache_creation"]
        )
        savings_percentage = (
            (self.token_counts["cache_read"] / total_tokens) * 100 if total_tokens > 0 else 0
        )
        print(
            f"Total input token savings from prompt caching: {savings_percentage:.2f}% of all input tokens used were read from cache."
        )
        print("Tokens read from cache come at a 90 percent discount!")

    # we use voyage AI here for embeddings. Read more here: https://docs.voyageai.com/docs/embeddings
    def _embed_and_store(self, texts: list[str], data: list[dict[str, Any]]):
        batch_size = 128
        result = [
            self.voyage_client.embed(texts[i : i + batch_size], model="voyage-2").embeddings
            for i in range(0, len(texts), batch_size)
        ]
        self.embeddings = [embedding for batch in result for embedding in batch]
        self.metadata = data

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.voyage_client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]

        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)
        return top_results

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError(
                "Vector database file not found. Use load_data to create a new database."
            )
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])
```

```python
# Load the transformed dataset
with open("data/codebase_chunks.json") as f:
    transformed_dataset = json.load(f)

# Initialize the ContextualVectorDB
contextual_db = ContextualVectorDB("my_contextual_db")

# Load and process the data
# note: consider increasing the number of parallel threads to run this faster, or reducing the number of parallel threads if concerned about hitting your API rate limit
contextual_db.load_data(transformed_dataset, parallel_threads=5)
```

These numbers reveal the power of prompt caching for contextual embeddings:

- We processed **737 chunks** across 9 codebase files
- **61.83% of input tokens** were read from cache (2.27M tokens at 90% discount)
- Without caching, this would cost **~$9.20** in input tokens
- With caching, the actual cost drops to **~$2.85** (69% savings)

The cache hit rate depends on how many chunks each document contains. Files with more chunks benefit more from caching since we write the full document to cache once, then read it repeatedly for each chunk in that file. This is why processing documents sequentially (rather than randomly shuffling chunks) is crucial for maximizing cache efficiency.

Now let's evaluate how much this contextualization improves our retrieval performance compared to the baseline.

```python
results = evaluate_and_display(
    contextual_db,
    "data/evaluation_set.jsonl",
    k_values=[5, 10, 20],
    db_name="Contextual Embeddings",
)
```

By adding context to each chunk before embedding, we've reduced retrieval failures by **~30-40%** across all k values. This means fewer irrelevant results in your top retrieved chunks, leading to better answers when you pass these chunks to Claude for final response generation.

The improvement is most pronounced at Pass@5, where precision matters most—suggesting that contextualized chunks aren't just retrieved more often, but rank higher when relevant.

### Contextual BM25: Hybrid Search

Contextual embeddings alone improved our Pass@10 from 87% to 92%. We can push performance even higher by combining semantic search with keyword-based search using **Contextual BM25**—a hybrid approach that reduces retrieval failure rates further.

#### Why Hybrid Search?

Semantic search excels at understanding meaning and context, but can miss exact keyword matches. BM25 (a probabilistic keyword ranking algorithm) excels at finding specific terms, but lacks semantic understanding. By combining both, we get the best of both worlds:

- **Semantic search**: Captures conceptual similarity and paraphrases
- **BM25**: Catches exact terminology, function names, and specific phrases
- **Reciprocal Rank Fusion**: Intelligently merges results from both sources

#### What is BM25?

BM25 is a probabilistic ranking function that improves upon TF-IDF by accounting for document length and term saturation. It's widely used in production search engines (including Elasticsearch) for its effectiveness at ranking keyword relevance. For technical details, see [this blog post](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables).

Instead of only searching the raw chunk content, we search both the chunk *and* the contextual description we generated earlier. This means BM25 can match keywords in either the original text or the explanatory context.

#### Setup: Running Elasticsearch

Before running the code below, you'll need Elasticsearch running locally. The easiest way is with Docker:

```bash
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:9.2.0
```

### Troubleshooting:
- Verify it's running: docker ps | grep elasticsearch
- If port 9200 is in use: docker stop elasticsearch && docker rm elasticsearch
- Check logs if issues occur: docker logs elasticsearch

### How the Hybrid Search Works

The retrieve_advanced function below implements a three-step process:

1. Retrieve candidates: Get top 150 results from both semantic search and BM25
2. Score fusion: Combine rankings using weighted Reciprocal Rank Fusion
    - Default: 80% weight to semantic search, 20% to BM25
    - These weights are tunable—experiment to optimize for your use case
3. Return top-k: Select the highest-scoring results after fusion

The weighting system lets you balance between semantic understanding and keyword precision based on your data characteristics.

```python
import json
import os
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm


class ElasticsearchBM25:
    def __init__(self, index_name: str = "contextual_bm25_index"):
        self.es_client = Elasticsearch("http://localhost:9200")
        self.index_name = index_name
        self.create_index()

    def create_index(self):
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False,
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "keyword", "index": False},
                    "chunk_id": {"type": "keyword", "index": False},
                    "original_index": {"type": "integer", "index": False},
                }
            },
        }

        # Change this line - remove 'body=' parameter
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(
                index=self.index_name,
                settings=index_settings["settings"],
                mappings=index_settings["mappings"],
            )
            print(f"Created index: {self.index_name}")

    def index_documents(self, documents: list[dict[str, Any]]):
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "content": doc["original_content"],
                    "contextualized_content": doc["contextualized_content"],
                    "doc_id": doc["doc_id"],
                    "chunk_id": doc["chunk_id"],
                    "original_index": doc["original_index"],
                },
            }
            for doc in documents
        ]
        success, _ = bulk(self.es_client, actions)
        self.es_client.indices.refresh(index=self.index_name)
        return success

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        self.es_client.indices.refresh(index=self.index_name)

        # Change this - remove 'body=' and pass query directly
        response = self.es_client.search(
            index=self.index_name,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                }
            },
            size=k,
        )

        return [
            {
                "doc_id": hit["_source"]["doc_id"],
                "original_index": hit["_source"]["original_index"],
                "content": hit["_source"]["content"],
                "contextualized_content": hit["_source"]["contextualized_content"],
                "score": hit["_score"],
            }
            for hit in response["hits"]["hits"]
        ]


def create_elasticsearch_bm25_index(db: ContextualVectorDB):
    es_bm25 = ElasticsearchBM25()
    es_bm25.index_documents(db.metadata)
    return es_bm25


def retrieve_advanced(
    query: str,
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    semantic_weight: float = 0.8,
    bm25_weight: float = 0.2,
):
    num_chunks_to_recall = 150

    # Semantic search
    semantic_results = db.search(query, k=num_chunks_to_recall)
    ranked_chunk_ids = [
        (result["metadata"]["doc_id"], result["metadata"]["original_index"])
        for result in semantic_results
    ]

    # BM25 search using Elasticsearch
    bm25_results = es_bm25.search(query, k=num_chunks_to_recall)
    ranked_bm25_chunk_ids = [
        (result["doc_id"], result["original_index"]) for result in bm25_results
    ]

    # Combine results
    chunk_ids = list(set(ranked_chunk_ids + ranked_bm25_chunk_ids))
    chunk_id_to_score = {}

    # Initial scoring with weights
    for chunk_id in chunk_ids:
        score = 0
        if chunk_id in ranked_chunk_ids:
            index = ranked_chunk_ids.index(chunk_id)
            score += semantic_weight * (1 / (index + 1))  # Weighted 1/n scoring for semantic
        if chunk_id in ranked_bm25_chunk_ids:
            index = ranked_bm25_chunk_ids.index(chunk_id)
            score += bm25_weight * (1 / (index + 1))  # Weighted 1/n scoring for BM25
        chunk_id_to_score[chunk_id] = score

    # Sort chunk IDs by their scores in descending order
    sorted_chunk_ids = sorted(
        chunk_id_to_score.keys(), key=lambda x: (chunk_id_to_score[x], x[0], x[1]), reverse=True
    )

    # Assign new scores based on the sorted order
    for index, chunk_id in enumerate(sorted_chunk_ids):
        chunk_id_to_score[chunk_id] = 1 / (index + 1)

    # Prepare the final results
    final_results = []
    semantic_count = 0
    bm25_count = 0
    for chunk_id in sorted_chunk_ids[:k]:
        chunk_metadata = next(
            chunk
            for chunk in db.metadata
            if chunk["doc_id"] == chunk_id[0] and chunk["original_index"] == chunk_id[1]
        )
        is_from_semantic = chunk_id in ranked_chunk_ids
        is_from_bm25 = chunk_id in ranked_bm25_chunk_ids
        final_results.append(
            {
                "chunk": chunk_metadata,
                "score": chunk_id_to_score[chunk_id],
                "from_semantic": is_from_semantic,
                "from_bm25": is_from_bm25,
            }
        )

        if is_from_semantic and not is_from_bm25:
            semantic_count += 1
        elif is_from_bm25 and not is_from_semantic:
            bm25_count += 1
        else:  # it's in both
            semantic_count += 0.5
            bm25_count += 0.5

    return final_results, semantic_count, bm25_count


def evaluate_db_advanced(
    db: ContextualVectorDB,
    original_jsonl_path: str,
    k_values: list[int] = None,
    db_name: str = "Hybrid Search",
):
    """
    Evaluate hybrid search (semantic + BM25) at multiple k values with formatted results.

    Args:
        db: ContextualVectorDB instance
        original_jsonl_path: Path to evaluation dataset
        k_values: List of k values to evaluate (default: [5, 10, 20])
        db_name: Name for the evaluation display

    Returns:
        Dict mapping k values to their results and source breakdowns
    """
    if k_values is None:
        k_values = [5, 10, 20]
    original_data = load_jsonl(original_jsonl_path)
    es_bm25 = create_elasticsearch_bm25_index(db)
    results = {}

    print(f"{'=' * 70}")
    print(f"Evaluation Results: {db_name}")
    print(f"{'=' * 70}\n")

    try:
        # Warm-up queries
        warm_up_queries = original_data[:10]
        for query_item in warm_up_queries:
            _ = retrieve_advanced(query_item["query"], db, es_bm25, k_values[0])

        for k in k_values:
            print(f"Evaluating Pass@{k}...")

            total_score = 0
            total_semantic_count = 0
            total_bm25_count = 0
            total_results = 0

            for query_item in tqdm(original_data, desc=f"Pass@{k}"):
                query = query_item["query"]
                golden_chunk_uuids = query_item["golden_chunk_uuids"]

                golden_contents = []
                for doc_uuid, chunk_index in golden_chunk_uuids:
                    golden_doc = next(
                        (doc for doc in query_item["golden_documents"] if doc["uuid"] == doc_uuid),
                        None,
                    )
                    if golden_doc:
                        golden_chunk = next(
                            (
                                chunk
                                for chunk in golden_doc["chunks"]
                                if chunk["index"] == chunk_index
                            ),
                            None,
                        )
                        if golden_chunk:
                            golden_contents.append(golden_chunk["content"].strip())

                if not golden_contents:
                    continue

                retrieved_docs, semantic_count, bm25_count = retrieve_advanced(
                    query, db, es_bm25, k
                )

                chunks_found = 0
                for golden_content in golden_contents:
                    for doc in retrieved_docs[:k]:
                        retrieved_content = doc["chunk"]["original_content"].strip()
                        if retrieved_content == golden_content:
                            chunks_found += 1
                            break

                query_score = chunks_found / len(golden_contents)
                total_score += query_score

                total_semantic_count += semantic_count
                total_bm25_count += bm25_count
                total_results += len(retrieved_docs)

            total_queries = len(original_data)
            average_score = total_score / total_queries
            pass_at_n = average_score * 100

            semantic_percentage = (
                (total_semantic_count / total_results) * 100 if total_results > 0 else 0
            )
            bm25_percentage = (total_bm25_count / total_results) * 100 if total_results > 0 else 0

            results[k] = {
                "pass_at_n": pass_at_n,
                "average_score": average_score,
                "total_queries": total_queries,
                "semantic_percentage": semantic_percentage,
                "bm25_percentage": bm25_percentage,
            }

            print(f"Pass@{k}: {pass_at_n:.2f}%")
            print(f"Semantic: {semantic_percentage:.1f}% | BM25: {bm25_percentage:.1f}%\n")

        # Print summary table
        print(f"{'=' * 70}")
        print(f"{'Metric':<12} {'Pass Rate':<12} {'Score':<12} {'Semantic':<12} {'BM25':<12}")
        print(f"{'-' * 70}")
        for k in k_values:
            r = results[k]
            print(
                f"{'Pass@' + str(k):<12} {r['pass_at_n']:>10.2f}% {r['average_score']:>10.4f} "
                f"{r['semantic_percentage']:>10.1f}% {r['bm25_percentage']:>10.1f}%"
            )
        print(f"{'=' * 70}\n")

        return results

    finally:
        # Delete the Elasticsearch index
        if es_bm25.es_client.indices.exists(index=es_bm25.index_name):
            es_bm25.es_client.indices.delete(index=es_bm25.index_name)
            print(f"Deleted Elasticsearch index: {es_bm25.index_name}")
```

```python
results = evaluate_db_advanced(
    contextual_db,
    "data/evaluation_set.jsonl",
    k_values=[5, 10, 20],
    db_name="Contextual BM25 Hybrid Search",
)
```

### Reranking

We've achieved strong results with hybrid search (93.21% Pass@10), but there's one more technique that can squeeze out additional performance: **reranking**.

#### What is Reranking?

Reranking is a two-stage retrieval approach:

1. **Stage 1 - Broad Retrieval**: Cast a wide net by retrieving more candidates than you need (e.g., retrieve 100 chunks)
2. **Stage 2 - Precise Selection**: Use a specialized reranking model to score these candidates and select only the top-k most relevant ones

**Why does this work?** Initial retrieval methods (embeddings, BM25) are optimized for speed across millions of documents. Reranking models are slower but more accurate—they can afford to do deeper analysis on a smaller candidate set. This creates a speed/accuracy trade-off that works well in practice.

#### Our Reranking Approach

For this example, we'll use a simpler reranking pipeline that builds on contextual embeddings alone (not the full hybrid search). Here's the process:

1. **Over-retrieve**: Get 10x more results than needed (e.g., retrieve 100 chunks when we need 10)
2. **Rerank with Cohere**: Use Cohere's `rerank-english-v3.0` model to score all candidates
3. **Select top-k**: Return only the highest-scoring results

The reranking model has access to both the original chunk content and the contextual descriptions we generated, giving it rich information to make precise relevance judgments.

#### Expected Performance

Adding reranking delivers a modest but meaningful improvement:
- **Without reranking**: 92.34% Pass@10 (contextual embeddings alone)
- **With reranking**: ~95% Pass@10 (additional 2-3% gain)

This might seem small, but in production systems, reducing failures from 7.66% to ~5% can significantly improve user experience. The trade-off is query latency—reranking adds ~100-200ms per query depending on candidate set size.

```python
import json
from collections.abc import Callable
from typing import Any

import cohere
from tqdm import tqdm


def evaluate_db_rerank(
    db, original_jsonl_path: str, k_values: list[int] = None, db_name: str = "Reranking"
):
    """
    Evaluate reranking performance at multiple k values with formatted results.

    Args:
        db: ContextualVectorDB instance
        original_jsonl_path: Path to evaluation dataset
        k_values: List of k values to evaluate (default: [5, 10, 20])
        db_name: Name for the evaluation display

    Returns:
        Dict mapping k values to their results
    """
    if k_values is None:
        k_values = [5, 10, 20]
    original_data = load_jsonl(original_jsonl_path)
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    results = {}

    print(f"{'=' * 60}")
    print(f"Evaluation Results: {db_name}")
    print(f"{'=' * 60}\n")

    for k in k_values:
        print(f"Evaluating Pass@{k} with reranking...")

        total_score = 0
        total_queries = len(original_data)

        for query_item in tqdm(original_data, desc=f"Pass@{k}"):
            query = query_item["query"]
            golden_chunk_uuids = query_item["golden_chunk_uuids"]

            # Find golden contents
            golden_contents = []
            for doc_uuid, chunk_index in golden_chunk_uuids:
                golden_doc = next(
                    (doc for doc in query_item["golden_documents"] if doc["uuid"] == doc_uuid), None
                )
                if golden_doc:
                    golden_chunk = next(
                        (chunk for chunk in golden_doc["chunks"] if chunk["index"] == chunk_index),
                        None,
                    )
                    if golden_chunk:
                        golden_contents.append(golden_chunk["content"].strip())

            if not golden_contents:
                continue

            # Retrieve and rerank
            semantic_results = db.search(query, k=k * 10)

            # Prepare documents for reranking
            documents = [
                f"{res['metadata']['original_content']}\n\nContext: {res['metadata']['contextualized_content']}"
                for res in semantic_results
            ]

            # Rerank
            rerank_response = co.rerank(
                model="rerank-english-v3.0", query=query, documents=documents, top_n=k
            )
            time.sleep(0.1)  # Rate limiting

            # Get final results
            retrieved_docs = []
            for r in rerank_response.results:
                original_result = semantic_results[r.index]
                retrieved_docs.append(
                    {"chunk": original_result["metadata"], "score": r.relevance_score}
                )

            # Check if golden chunks are in results
            chunks_found = 0
            for golden_content in golden_contents:
                for doc in retrieved_docs[:k]:
                    retrieved_content = doc["chunk"]["original_content"].strip()
                    if retrieved_content == golden_content:
                        chunks_found += 1
                        break

            query_score = chunks_found / len(golden_contents)
            total_score += query_score

        average_score = total_score / total_queries
        pass_at_n = average_score * 100

        results[k] = {
            "pass_at_n": pass_at_n,
            "average_score": average_score,
            "total_queries": total_queries,
        }

        print(f"Pass@{k}: {pass_at_n:.2f}%")
        print(f"Average Score: {average_score:.4f}\n")

    # Print summary table
    print(f"{'=' * 60}")
    print(f"{'Metric':<15} {'Pass Rate':<15} {'Score':<15}")
    print(f"{'-' * 60}")
    for k in k_values:
        pass_rate = f"{results[k]['pass_at_n']:.2f}%"
        score = f"{results[k]['average_score']:.4f}"
        print(f"{'Pass@' + str(k):<15} {pass_rate:<15} {score:<15}")
    print(f"{'=' * 60}\n")

    return results
```

```python
results = evaluate_db_rerank(
    contextual_db,
    "data/evaluation_set.jsonl",
    k_values=[5, 10, 20],
    db_name="Contextual Embeddings + Reranking",
)
```


Reranking delivers our strongest results, nearly eliminating retrieval failures. Let's look at how each technique built upon the previous one to achieve this improvement.

Starting from our baseline RAG system at 87% Pass@10, we've climbed to over 95% by systematically applying advanced retrieval techniques. Each method addresses a different weakness: contextual embeddings solve the "isolated chunk" problem, hybrid search catches keyword-specific queries that embeddings miss, and reranking applies more sophisticated relevance scoring to refine the final selection.

| Approach | Pass@5 | Pass@10 | Pass@20 |
|----------|--------|---------|---------|
| **Baseline RAG** | 80.92% | 87.15% | 90.06% |
| **+ Contextual Embeddings** | 88.12% | 92.34% | 94.29% |
| **+ Hybrid Search (BM25)** | 86.43% | 93.21% | 94.99% |
| **+ Reranking** | 92.15% | 95.26% | 97.45% |

**Key Takeaways:**

1. **Contextual embeddings provided the largest single improvement** (+5-7 percentage points), validating that adding document-level context to chunks significantly improves retrieval quality. This technique alone gets you 90% of the way to optimal performance.

2. **Reranking achieves the highest absolute performance**, reaching 95.26% Pass@10—meaning the correct chunk appears in the top 10 results for 95% of queries. This represents a **47% reduction in retrieval failures** compared to baseline RAG (from 12.85% failure rate down to 4.74%).

3. **Trade-offs matter**: Each technique adds complexity and cost:
   - Contextual embeddings: One-time ingestion cost (~$3 for this dataset with prompt caching)
   - Hybrid search: Requires Elasticsearch infrastructure and maintenance
   - Reranking: Adds 100-200ms query latency and per-query API costs (~$0.002 per query)

4. **Choose your approach** based on your requirements:
   - **High-volume, cost-sensitive**: Contextual embeddings alone (92% Pass@10, no per-query costs)
   - **Maximum accuracy, latency-tolerant**: Full reranking pipeline (95% Pass@10, best precision)
   - **Balanced production system**: Hybrid search for strong performance without per-query costs (93% Pass@10)

For most production RAG systems, **contextual embeddings provide the best performance-to-cost ratio**, delivering 92% Pass@10 with only one-time ingestion costs. Hybrid search and reranking are available when you need that extra 2-3 percentage points of precision and can afford the additional infrastructure or query costs.

#### Next Steps and Key Takeaways

1) We demonstrated how to use Contextual Embeddings to improve retrieval performance, then delivered additional improvements with Contextual BM25 and reranking.

2) This example used codebases, but these methods also apply to other data types such as internal company knowledge bases, financial & legal content, educational content, and much more. 

3) If you are an AWS user, you can get started with the Lambda function in `contextual-rag-lambda-function`, and if you're a GCP user you can spin up your own Cloud Run instance and follow a similar pattern!
