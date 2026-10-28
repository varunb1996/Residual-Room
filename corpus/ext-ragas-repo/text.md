# Ragas docs

# docs/concepts/metrics/available_metrics/faithfulness.md

### Faithfulness

The **Faithfulness** metric measures how factually consistent a `response` is with the `retrieved context`. It ranges from 0 to 1, with higher scores indicating better consistency.

A response is considered **faithful** if all its claims can be supported by the retrieved context.

To calculate this:
1. Identify all the claims in the response.
2. Check each claim to see if it can be inferred from the retrieved context.
3. Compute the faithfulness score using the formula:

$$
\text{Faithfulness Score} = \frac{\text{Number of claims in the response supported by the retrieved context}}{\text{Total number of claims in the response}}
$$


#### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = Faithfulness(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ]
)
print(f"Faithfulness Score: {result.value}")
```

Output:

```
Faithfulness Score: 1.0
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[...]
    )
    ```


#### How It’s Calculated

!!! example
    **Question**: Where and when was Einstein born?

    **Context**: Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time

    **High faithfulness answer**: Einstein was born in Germany on 14th March 1879.

    **Low faithfulness answer**:  Einstein was born in Germany on 20th March 1879.

Let's examine how faithfulness was calculated using the low faithfulness answer:

- **Step 1:** Break the generated answer into individual statements.
    - Statements:
        - Statement 1: "Einstein was born in Germany."
        - Statement 2: "Einstein was born on 20th March 1879."

- **Step 2:** For each of the generated statements, verify if it can be inferred from the given context.
    - Statement 1: Yes
    - Statement 2: No

- **Step 3:** Use the formula depicted above to calculate faithfulness.

    $$
    \text{Faithfulness} = { \text{1} \over \text{2} } = 0.5
    $$


### Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

#### Example with SingleTurnSample

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )
scorer = Faithfulness(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

Output:

```
1.0
```

#### Faithfulness with HHEM-2.1-Open

[Vectara's HHEM-2.1-Open](https://vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model/) is a classifier model (T5) that is trained to detect hallucinations from LLM generated text. This model can be used in the second step of calculating faithfulness, i.e. when claims are cross-checked with the given context to determine if it can be inferred from the context. The model is free, small, and open-source, making it very efficient in production use cases.

To use the model to calculate faithfulness:

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import FaithfulnesswithHHEM


sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )
scorer = FaithfulnesswithHHEM(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

You can load the model onto a specified device by setting the `device` argument and adjust the batch size for inference using the `batch_size` parameter. By default, the model is loaded on the CPU with a batch size of 10:

```python
my_device = "cuda:0"
my_batch_size = 10

scorer = FaithfulnesswithHHEM(device=my_device, batch_size=my_batch_size)
await scorer.single_turn_ascore(sample)
```

# docs/concepts/metrics/available_metrics/context_precision.md

## Context Precision

Context Precision is a metric that evaluates the retriever's ability to rank relevant chunks higher than irrelevant ones for a given query in the retrieved context. Specifically, it assesses the degree to which relevant chunks in the retrieved context are placed at the top of the ranking.

It is calculated as the mean of the precision@k for each chunk in the context. Precision@k is the ratio of the number of relevant chunks at rank k to the total number of chunks at rank k.

$$
\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{Total number of relevant items in the top } K \text{ results}}
$$

$$
\text{Precision@k} = {\text{true positives@k} \over  (\text{true positives@k} + \text{false positives@k})}
$$

Where $K$ is the total number of chunks in `retrieved_contexts` and $v_k \in \{0, 1\}$ is the relevance indicator at rank $k$.

### Examples

#### Context Precision

The `ContextPrecision` metric evaluates whether retrieved contexts are useful for answering a question by comparing each context against a reference answer. Use this when you have a reference answer available.

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextPrecision(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[
        "The Eiffel Tower is located in Paris.",
        "The Brandenburg Gate is located in Berlin."
    ]
)
print(f"Context Precision Score: {result.value}")
```

Output:
```
Context Precision Score: 0.9999999999
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=[...]
    )
    ```

#### Context Utilization

The `ContextUtilization` metric evaluates whether retrieved contexts are useful by comparing each context against the generated response. Use this when you don't have a reference answer but have the response that was generated.

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextUtilization

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextUtilization(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[
        "The Eiffel Tower is located in Paris.",
        "The Brandenburg Gate is located in Berlin."
    ]
)
print(f"Context Utilization Score: {result.value}")
```

Output:
```
Context Utilization Score: 0.9999999999
```

Note that even if an irrelevant chunk is present at the second position in the array, context precision remains the same. However, if this irrelevant chunk is placed at the first position, context precision reduces:

```python
result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[
        "The Brandenburg Gate is located in Berlin.",
        "The Eiffel Tower is located in Paris."
    ]
)
print(f"Context Utilization Score: {result.value}")
```

Output:
```
Context Utilization Score: 0.49999999995
```

### Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

#### Example with SingleTurnSample

```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference

context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
)

await context_precision.single_turn_ascore(sample)
```

Output:
```
0.9999999999
```

#### Context Precision without reference

The `LLMContextPrecisionWithoutReference` metric can be used without the availability of a reference answer. To estimate if the retrieved contexts are relevant, this method uses the LLM to compare each chunk in `retrieved_contexts` with the `response`.

##### Example

```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference

context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
)

await context_precision.single_turn_ascore(sample)
```

Output:
```
0.9999999999
```

#### Context Precision with reference

The `LLMContextPrecisionWithReference` metric can be used when you have both retrieved contexts and also a reference response associated with a `user_input`. To estimate if the retrieved contexts are relevant, this method uses the LLM to compare each chunk in `retrieved_contexts` with the `reference`.

##### Example

```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference

context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
)

await context_precision.single_turn_ascore(sample)
```

Output:
```
0.9999999999
```

### Non LLM Based Context Precision

This metric uses non-LLM-based methods (such as [Levenshtein distance measure](https://en.wikipedia.org/wiki/Levenshtein_distance)) to determine whether a retrieved context is relevant.

#### Context Precision with reference contexts

The `NonLLMContextPrecisionWithReference` metric is designed for scenarios where both retrieved contexts and reference contexts are available for a `user_input`. To determine if a retrieved context is relevant, this method compares each retrieved context or chunk in `retrieved_contexts` with every context in `reference_contexts` using a non-LLM-based similarity measure.

Note that this metric would need the rapidfuzz package to be installed: `pip install rapidfuzz`.

##### Example

```python
from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference

context_precision = NonLLMContextPrecisionWithReference()

sample = SingleTurnSample(
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
    reference_contexts=["Paris is the capital of France.", "The Eiffel Tower is one of the most famous landmarks in Paris."]
)

await context_precision.single_turn_ascore(sample)
```

Output:
```
0.9999999999
```

### ID Based Context Precision

IDBasedContextPrecision provides a direct and efficient way to measure precision by comparing the IDs of retrieved contexts with reference context IDs. This metric is particularly useful when you have a unique ID system for your documents and want to evaluate retrieval performance without comparing the actual content.

The metric computes precision using retrieved_context_ids and reference_context_ids, with values ranging between 0 and 1. Higher values indicate better performance. It works with both string and integer IDs.

The formula for calculating ID-based context precision is as follows:

$$ \text{ID-Based Context Precision} = \frac{\text{Number of retrieved context IDs found in reference context IDs}}{\text{Total number of retrieved context IDs}} $$

#### Example

```python
from ragas import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision

sample = SingleTurnSample(
    retrieved_context_ids=["doc_1", "doc_2", "doc_3", "doc_4"],
    reference_context_ids=["doc_1", "doc_4", "doc_5", "doc_6"]
)

id_precision = IDBasedContextPrecision()
await id_precision.single_turn_ascore(sample)
```

Output:
```
0.5
```

In this example, out of the 4 retrieved context IDs, only 2 ("doc_1" and "doc_4") are found in the reference context IDs, resulting in a precision score of 0.5 or 50%.

# docs/concepts/metrics/available_metrics/context_recall.md

## Context Recall

Context Recall measures how many of the relevant documents (or pieces of information) were successfully retrieved. It focuses on not missing important results. Higher recall means fewer relevant documents were left out. In short, recall is about not missing anything important. 

Since it is about not missing anything, calculating context recall always requires a reference to compare against. The LLM-based Context Recall metric uses `reference` as a proxy to `reference_contexts`, which makes it easier to use as annotating reference contexts can be very time-consuming. To estimate context recall from the `reference`, the reference is broken down into claims, and each claim is analyzed to determine whether it can be attributed to the retrieved context or not. In an ideal scenario, all claims in the reference answer should be attributable to the retrieved context.

The formula for calculating context recall is as follows:

$$
\text{Context Recall} = \frac{\text{Number of claims in the reference supported by the retrieved context}}{\text{Total number of claims in the reference}}
$$

### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextRecall(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    retrieved_contexts=["Paris is the capital of France."],
    reference="The Eiffel Tower is located in Paris."
)
print(f"Context Recall Score: {result.value}")
```

Output:

```
Context Recall Score: 1.0
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        user_input="Where is the Eiffel Tower located?",
        retrieved_contexts=["Paris is the capital of France."],
        reference="The Eiffel Tower is located in Paris."
    )
    ```

### LLM Based Context Recall (Legacy API)

!!! warning "Legacy API"
    The following example uses the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above. This API will be deprecated in version 0.4 and removed in version 1.0.

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import LLMContextRecall

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["Paris is the capital of France."],
)

context_recall = LLMContextRecall(llm=evaluator_llm)
await context_recall.single_turn_ascore(sample)
```

Output:
```
1.0
```

### Non LLM Based Context Recall

`NonLLMContextRecall` metric is computed using `retrieved_contexts` and `reference_contexts`, and the values range between 0 and 1, with higher values indicating better performance. This metrics uses non-LLM string comparison metrics to identify if a retrieved context is relevant or not. You can use any non LLM based metrics as distance measure to identify if a retrieved context is relevant or not.

The formula for calculating context recall is as follows:

$$
\text{context recall} = {|\text{Number of relevant contexts retrieved}| \over |\text{Total number of reference contexts}|}
$$

#### Example

```python


from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

sample = SingleTurnSample(
    retrieved_contexts=["Paris is the capital of France."],
    reference_contexts=["Paris is the capital of France.", "The Eiffel Tower is one of the most famous landmarks in Paris."]
)

context_recall = NonLLMContextRecall()
await context_recall.single_turn_ascore(sample)


```
Output
```
0.5
```

### ID BasedContext Recall

ID Based Context Recall
IDBasedContextRecall provides a direct and efficient way to measure recall by comparing the IDs of retrieved contexts with reference context IDs. This metric is particularly useful when you have a unique ID system for your documents and want to evaluate retrieval performance without comparing the actual content.

The metric computes recall using retrieved_context_ids and reference_context_ids, with values ranging between 0 and 1. Higher values indicate better performance. It works with both string and integer IDs.

The formula for calculating ID-based context recall is as follows:

$$ \text{ID-Based Context Recall} = \frac{\text{Number of reference context IDs found in retrieved context IDs}}{\text{Total number of reference context IDs}} $$

#### Example

```python

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextRecall

sample = SingleTurnSample(
    retrieved_context_ids=["doc_1", "doc_2", "doc_3"], 
    reference_context_ids=["doc_1", "doc_4", "doc_5", "doc_6"]
)

id_recall = IDBasedContextRecall()
await id_recall.single_turn_ascore(sample)
```

Output
```
0.25
```

# docs/concepts/metrics/overview/index.md

## Overview of Metrics

### Why Metrics Matter

You can't improve what you don't measure. Metrics are the feedback loop that makes iteration possible.

In AI systems, progress depends on running many experiments—each a hypothesis about how to improve performance. But without a clear, reliable metric, you can't tell the difference between a successful experiment (a positive delta between the new score and the old one) and a failed one.

Metrics give you a compass. They let you quantify improvement, detect regressions, and align optimization efforts with user impact and business value.

A metric is a quantitative measure used to evaluate the performance of a AI application. Metrics help in assessing how well the application and individual components that makes up application is performing relative to the given test data. They provide a numerical basis for comparison, optimization, and decision-making throughout the application development and deployment process. Metrics are crucial for:

1. **Component Selection**: Metrics can be used to compare different components of the AI application like LLM, Retriever, Agent configuration, etc with your own data and select the best one from different options.
2. **Error Diagnosis and Debugging**: Metrics help identify which part of the application is causing errors or suboptimal performance, making it easier to debug and refine.
3. **Continuous Monitoring and Maintenance**: Metrics enable the tracking of an AI application's performance over time, helping to detect and respond to issues such as data drift, model degradation, or changing user requirements.

### Types of Metrics in AI Applications

#### 1. End-to-End Metrics

End-to-end metrics evaluate the overall system performance from the user's perspective, treating the AI application as a black box. These metrics quantify key outcomes users care deeply about, based solely on the system's final outputs.

Examples:

- Answer correctness: Measures if the provided answers from a Retrieval-Augmented Generation (RAG) system are accurate.
- Citation accuracy: Evaluates whether the references cited by the RAG system are correctly identified and relevant.

Optimizing end-to-end metrics ensures tangible improvements aligned directly with user expectations.

#### 2. Component-Level Metrics

Component-level metrics assess the individual parts of an AI system independently. These metrics are immediately actionable and facilitate targeted improvements but do not necessarily correlate directly with end-user satisfaction.

Example:

- Retrieval accuracy: Measures how effectively a RAG system retrieves relevant information. A low retrieval accuracy (e.g., 50%) signals that improving this component can enhance overall system performance. However, improving a component alone doesn't guarantee better end-to-end outcomes.

#### 3. Business Metrics

Business metrics align AI system performance with organizational objectives and quantify tangible business outcomes. These metrics are typically lagging indicators, calculated after a deployment period (days/weeks/months).

Example:

- Ticket deflection rate: Measures the percentage reduction of support tickets due to the deployment of an AI assistant.

### Types of Metrics in Ragas

<figure markdown="span">
  ![Component-wise Evaluation](../../../_static/imgs/metrics_mindmap.png){width="600"}
  <figcaption>Metrics Mind map</figcaption>
</figure>

**Metrics can be classified into two categories based on the mechanism used underneath the hood**:

&nbsp;&nbsp;&nbsp;&nbsp; **LLM-based metrics**: These metrics use LLM underneath to do the evaluation. There might be one or more LLM calls that are performed to arrive at the score or result. These metrics can be somewhat non-deterministic as the LLM might not always return the same result for the same input. On the other hand, these metrics has shown to be more accurate and closer to human evaluation.

All LLM based metrics in ragas are inherited from `MetricWithLLM` class. These metrics expects a LLM object to be set before scoring.

```python
from ragas.metrics import FactualCorrectness
scorer = FactualCorrectness(llm=evaluation_llm)
```

Each LLM based metrics also will have prompts associated with it written using [Prompt Object](./../../components/prompt.md). You can customize these prompts to suit your domain and use-case. Learn more in the [Modifying Prompts in Metrics](../../../howtos/customizations/metrics/modifying-prompts-metrics.md) guide.


&nbsp;&nbsp;&nbsp;&nbsp; **Non-LLM-based metrics**: These metrics do not use LLM underneath to do the evaluation. These metrics are deterministic and can be used to evaluate the performance of the AI application without using LLM. These metrics rely on traditional methods to evaluate the performance of the AI application, such as string similarity, BLEU score, etc. Due to the same, these metrics are known to have a lower correlation with human evaluation.

All Non-LLM-based metrics in ragas are inherited from `Metric` class.

**Metrics can be broadly classified into two categories based on the type of data they evaluate**:

&nbsp;&nbsp;&nbsp;&nbsp; **Single turn metrics**: These metrics evaluate the performance of the AI application based on a single turn of interaction between the user and the AI. All metrics in ragas that supports single turn evaluation are inherited from [SingleTurnMetric][ragas.metrics.base.SingleTurnMetric] class and scored using `single_turn_ascore` method. It also expects a [Single Turn Sample][ragas.dataset_schema.SingleTurnSample] object as input.

```python
from ragas.metrics import FactualCorrectness

scorer = FactualCorrectness()
await scorer.single_turn_ascore(sample)
```

&nbsp;&nbsp;&nbsp;&nbsp; **Multi-turn metrics**: These metrics evaluate the performance of the AI application based on multiple turns of interaction between the user and the AI. All metrics in ragas that supports multi turn evaluation are inherited from [MultiTurnMetric][ragas.metrics.base.MultiTurnMetric] class and scored using `multi_turn_ascore` method. It also expects a [Multi Turn Sample][ragas.dataset_schema.MultiTurnSample] object as input.

```python
from ragas.metrics import AgentGoalAccuracy
from ragas import MultiTurnSample

scorer = AgentGoalAccuracy()
await scorer.multi_turn_ascore(sample)
```

#### Output Types

In Ragas, we categorize metrics based on the type of output they produce. This classification helps clarify how each metric behaves and how its results can be interpreted or aggregated. The three types are:

##### 1. Discrete Metrics

These return a single value from a predefined list of categorical classes. There is no implicit ordering among the classes. Common use cases include classifying outputs into categories such as pass/fail or good/okay/bad. Discrete metrics accept custom prompts directly, making them ideal for quick custom evaluations.

Example:
```python
from ragas.metrics import discrete_metric

@discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
def my_metric(predicted: str, expected: str) -> str:
    return "pass" if predicted.lower() == expected.lower() else "fail"
```

For modifying prompts in existing collection metrics (like Faithfulness, FactualCorrectness), see [Modifying prompts in metrics](../../../howtos/customizations/metrics/modifying-prompts-metrics.md).

##### 2. Numeric Metrics

These return an integer or float value within a specified range. Numeric metrics support aggregation functions such as mean, sum, or mode, making them useful for statistical analysis.

```python
from ragas.metrics import numeric_metric

@numeric_metric(name="response_accuracy", allowed_values=(0, 1))
def my_metric(predicted: float, expected: float) -> float:
    return abs(predicted - expected) / max(expected, 1e-5)

my_metric.score(predicted=0.8, expected=1.0)  # Returns a float value
```

##### 3. Ranking Metrics

These evaluate multiple outputs at once and return a ranked list based on a defined criterion. They are useful when the goal is to compare multiple outputs from the same pipeline relative to one another.

```python
from ragas.metrics import ranking_metric
@ranking_metric(name="response_ranking", allowed_values=[0,1])
def my_metric(responses: list) -> list:
    response_lengths = [len(response) for response in responses]
    sorted_indices = sorted(range(len(response_lengths)), key=lambda i: response_lengths[i])
    return sorted_indices

my_metric.score(responses=["short", "a bit longer", "the longest response"])  # Returns a ranked list of indices
```

### Metric Design Principles

Designing effective metrics for AI applications requires following to a set of core principles to ensure their reliability, interpretability, and relevance. Here are five key principles we follow in ragas when designing metrics:

**1. Single-Aspect Focus**
A single metric should target only one specific aspect of the AI application's performance. This ensures that the metric is both interpretable and actionable, providing clear insights into what is being measured.

**2. Intuitive and Interpretable**
Metrics should be designed to be easy to understand and interpret. Clear and intuitive metrics make it simpler to communicate results and draw meaningful conclusions.

**3. Effective Prompt Flows**
When developing metrics using large language models (LLMs), use intelligent prompt flows that align closely with human evaluation. Decomposing complex tasks into smaller sub-tasks with specific prompts can improve the accuracy and relevance of the metric.

**4. Robustness**
Ensure that LLM-based metrics include sufficient few-shot examples that reflect the desired outcomes. This enhances the robustness of the metric by providing context and guidance for the LLM to follow.

**5.Consistent Scoring Ranges**
It is crucial to normalize metric score values or ensure they fall within a specific range, such as 0 to 1. This facilitates comparison between different metrics and helps maintain consistency and interpretability across the evaluation framework.

These principles serve as a foundation for creating metrics that are not only effective but also practical and meaningful in evaluating AI applications.

### Choosing the Right Metrics for Your Application

#### 1. Prioritize End-to-End Metrics

Focus first on metrics reflecting overall user satisfaction. While many aspects influence user satisfaction—such as factual correctness, response tone, and explanation depth—concentrate initially on the few dimensions delivering maximum user value (e.g., answer and citation accuracy in a RAG-based assistant).

#### 2. Ensure Interpretability

Design metrics clear enough for the entire team to interpret and reason about. For example:

- Execution accuracy in a text-to-SQL system: Does the SQL query generated return precisely the same dataset as the ground truth query crafted by domain experts?

#### 3. Emphasize Objective Over Subjective Metrics

Prioritize metrics with objective criteria, minimizing subjective judgment. Assess objectivity by independently labeling samples across team members and measuring agreement levels. A high inter-rater agreement (≥80%) indicates greater objectivity.

#### 4. Few Strong Signals over Many Weak Signals

Avoid a proliferation of metrics that provide weak signals and impede clear decision-making. Instead, select fewer metrics offering strong, reliable signals. For instance:

- In a conversational AI, using a single metric such as goal accuracy (whether the user's objective for interacting with the AI was met) provides strong proxy for the performance of the system than multiple weak proxies like coherence or helpfulness.
