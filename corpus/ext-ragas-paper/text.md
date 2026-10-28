# RAGAS (Es et al., EACL 2024 demo)

Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics System Demonstrations, pages 150–158 March 17-22, 2024 c©2024 Association for Computational Linguistics RAGA S: Automated Evaluation of Retrieval Augmented Generation Shahul Es†, Jithin James †, Luis Espinosa-Anke ∗♢, Steven Schockaert ∗ †Exploding Gradients ∗CardiffNLP, Cardiff University, United Kingdom ♢AMPLYFI, United Kingdom {shahules786,jamesjithin97}@gmail.com {espinosa-ankel,schockaerts1}@cardiff.ac.uk

## Abstract

We introduce RAGA S1 (Retrieval Augmented Generation Assessment), a framework for reference-free evaluation of Retrieval Augmented Generation (RAG) pipelines. RAG systems are composed of a retrieval module and an LLM based generation module. They provide LLMs with knowledge from a reference corpus, which can help to keep LLM based systems up-to-date and can reduce the risk of hallucinations, among others. However, evaluating RAG architectures is challenging because there are several dimensions to consider: the ability of the retrieval system to identify relevant and focused context passages, the ability of the LLM to exploit such passages in a faithful way, and the quality of the generation itself. With RAGA S, we put forward a suite of metrics which can be used to evaluate these different dimensions without having to rely on ground truth human annotations. We posit that such a framework can crucially contribute to faster evaluation cycles of RAG architectures, which is especially important given the fast adoption of LLMs.

## 1 Introduction

Language Models (LMs) capture a vast amount of knowledge about the world, which allows them to answer questions without accessing any external sources. This idea of LMs as repositories of knowledge emerged shortly after the introduction of BERT (Devlin et al., 2019) and became more firmly established with the introduction of ever larger LMs (Roberts et al., 2020). While the most recent Large Language Models (LLMs) capture enough knowledge to rival human performance across a wide variety of question answering benchmarks (Bubeck et al., 2023), the idea of using LLMs as knowledge bases still has two fundamental limitations. First, LLMs are not able to answer 1RAGA S is available at https://github.com/ explodinggradients/ragas.

questions about events that have happened after they were trained. Second, even the largest models struggle to memorise knowledge that is only rarely mentioned in the training corpus (Kandpal et al., 2022; Mallen et al., 2023). The standard solution to these issues is to rely on Retrieval Augmented Generation (RAG) (Lee et al., 2019; Lewis et al., 2020; Guu et al., 2020). Answering a question then essentially involves retrieving relevant passages from a corpus and feeding these passages, along with the original question, to the LM. While initial approaches relied on specialised LMs for retrieval-augmented language modelling (Khandelwal et al., 2020; Borgeaud et al., 2022), recent work has suggested that simply adding retrieved documents to the input of a standard LM can also work well (Khattab et al., 2022; Ram et al., 2023; Shi et al., 2023), thus making it possible to use retrievalaugmented strategies in combination with LLMs that are only available through APIs.

While the usefulness of retrieval-augmented strategies is clear, their implementation requires a significant amount of tuning, as the overall performance will be affected by the retrieval model, the considered corpus, the LM and the prompt formulation, among others. Automated evaluation of retrieval-augmented systems is thus paramount. In practice, RAG systems are often evaluated in terms of the language modelling task itself, i.e. by measuring perplexity on some reference corpus. However, such evaluations are not always predictive of downstream performance (Wang et al., 2023c).

Moreover, this evaluation strategy relies on the LM probabilities, which are not accessible for some closed models (e.g. ChatGPT and GPT-4). Question answering is another common evaluation task, but usually only datasets with short extractive answers are considered, which may not be representative of how the system will be used.

To address these issues, in this paper we present RAGA S, a framework for the automated assess150 ment of retrieval augmented generation systems.

We focus on settings where reference answers may not be available, and where we want to estimate different proxies for correctness, in addition to the usefulness of the retrieved passages. The RAGA S framework provides an integration with both llamaindex and Langchain, the most widely used frameworks for building RAG solutions, thus enabling developers to easily integrate RAGA S into their standard workflow.

## 2 Related Work

Estimating faithfulness using LLMs The problem of detecting hallucinations in LLM-generated responses has been extensively studied (Ji et al., 2023). Some authors have suggested the idea of predicting factuality using a few-shot prompting strategy (Zhang et al., 2023). Recent analyses, however, suggest that existing models struggle with detecting hallucination when using standard prompting strategies (Li et al., 2023; Azaria and Mitchell, 2023). Other approaches rely on linking the generated responses to facts from an external knowledge base (Min et al., 2023), but this is not always possible.

Yet another strategy is to inspect the probabilities assigned to individual tokens, where we would expect the model to be less confident in hallucinated answers than in factual ones. For instance, BARTScore (Yuan et al., 2021) estimates factuality by looking at the conditional probability of the generated text given the input. Kadavath et al. (2022) use a variation of this idea. Starting from the observation that LLMs provide well-calibrated probabilities when answering multiple-choice questions, they essentially convert the problem of validating model generated answers into a multiple-choice question which asks whether the answer is true or false. Rather than looking at the output probabilities, Azaria and Mitchell (2023) propose to train a supervised classifier on the weights from one of the hidden layers of the LLM, to predict whether a given statement is true or not. While the approach performs well, the need to access the hidden states of the model makes it unsuitable for systems that access LLMs through an API.

For models that do not provide access to token probabilities, such as ChatGPT and GPT-4, different methods are needed. SelfCheckGPT (Manakul et al., 2023) addresses this problem by instead sampling multiple answers. Their core idea is that factual answers are more stable: when an answer is factual, we can expect that different samples will tend to be semantically similar, whereas this is less likely to be the case for hallucinated answers.

Automated evaluation of text generation systems LLMs have also been leveraged to automatically evaluate other aspects of generated text fragments, beyond factuality. For instance, GPTScore (Fu et al., 2023) uses a prompt that specifies the considered aspect (e.g. fluency) and then scores passages based on the average probability of the generated tokens, according to a given autoregressive LM.

This idea of using prompts was previously also considered by Yuan et al. (2021), although they used a smaller fine-tuned LM (i.e. BART) and did not observe a clear benefit from using prompts. Another approach directly asks ChatGPT to evaluate a particular aspect of the given answer by providing a score between 0 and 100, or by providing a rating on a 5-star scale (Wang et al., 2023a). Remarkably, strong results can be obtained in this way, although it comes with the limitation of being sensitive to the design of the prompt. Rather than scoring individual answers, some authors have also focused on using an LLM to select the best answer among a number of candidates (Wang et al., 2023b), typically to compare the performance of different LLMs. However, care is needed with this approach, as the order in which the answers are presented can influence the result (Wang et al., 2023b).

More generally, however, most approaches have relied on the availability of one or more reference answers for evaluating text generation systems. For instance, BERTScore (Zhang et al., 2020) and MoverScore (Zhao et al., 2019) use contextualised embeddings, produced by a pre-trained BERT model, to compare the similarity between the generated answer and the reference answers.

BARTScore (Yuan et al., 2021) similarly uses reference answers to compute aspects such as precision (estimated as the probability of generating the generated answer given the reference) and recall (estimated as the probability of generating the reference given the generated answer).

## 3 Evaluation Strategies

We consider a standard RAG setting, where given a question q, the system first retrieves some context c(q) and then uses the retrieved context to generate an answer as(q). When building a RAG system, we usually do not have access to human-annotated 151 datasets or reference answers. We therefore focus on metrics that are fully self-contained and reference-free. We focus in particular three quality aspects, which we argue are of central importance.

First, Faithfulness refers to the idea that the answer should be grounded in the given context.

This is important to avoid hallucinations, and to ensure that the retrieved context can act as a justification for the generated answer. Indeed, RAG systems are often used in applications where the factual consistency of the generated text w.r.t. the grounded sources is highly important, e.g. in domains such as law, where information is constantly evolving. Second, Answer Relevance refers to the idea that the generated answer should address the actual question that was provided. Finally, Context Relevance refers to the idea that the retrieved context should be focused, containing as little irrelevant information as possible. This is important given the cost associated with feeding long context passages to LLMs. Moreover, when context passages are too long, LLMs are often less effective in exploiting that context, especially for information that is provided in the middle of the context passage (Liu et al., 2023).

We now explain how these three quality aspects can be measured in a fully automated way, by prompting an LLM. In our implementation and experiments, all prompts are evaluated using the gpt-3.5-turbo-16k model, which is available through the OpenAI API2.

Faithfulness We say that the answer as(q) is faithful to the context c(q) if the claims that are made in the answer can be inferred from the context. To estimate faithfulness, we first use an LLM to extract a set of statements, S(as(q)). The aim of this step is to decompose longer sentences into shorter and more focused assertions. We use the following prompt for this step3:

Given a question and answer, create one or more statements from each sentence in the given answer.

question: [question] answer: [answer] where [question] and [answer] refer to the given question and answer. For each statementsi in 2https://platform.openai.com 3To help clarify the task, we include a demonstration as part of the prompt. This demonstration is not explicitly shown in the listing of the prompts throughout this paper.

S(as(q)), the LLM determines if si can be inferred from c(q) using a verification function v(si, c(q)).

This verification step is carried out using the following prompt:

Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

statement: [statement 1] ...

statement: [statement n] The final faithfulness score, F , is then computed as F = |V ||S| , where |V | is the number of statements that were supported according to the LLM and |S| is the total number of statements in S(as(q)).

Answer relevance We say that the answer as(q) is relevant if it directly addresses the question in an appropriate way. In particular, our assessment of answer relevance does not take into account factuality, but penalises cases where the answer is incomplete or where it contains redundant information. To estimate answer relevance, for the given answer as(q), we prompt the LLM to generate n potential questions qi based on as(q), as follows:

Generate a question for the given answer.

answer: [answer] We then obtain embeddings for all questions using the text-embedding-ada-002 model, available from the OpenAI API. For each qi, we calculate the similarity sim(q, qi) with the original question q, as the cosine between the corresponding embeddings. The answer relevance score, AR, for question q is then computed as:

AR = 1n n∑i=1 sim(q, qi) (1) This metric evaluates how closely the generated answer aligns with the initial question or instruction.

152 Context relevance The context c(q) is considered relevant to the extent that it exclusively contains information that is needed to answer the question. In particular, this metric aims to penalise the inclusion of redundant information. To estimate context relevance, given a question q and its context c(q), the LLM extracts a subset of sentences, Sext, from c(q) that are crucial to answer q, using the following prompt:

Please extract relevant sentences from the provided context that can potentially help answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information". While extracting candidate sentences you’re not allowed to make any changes to sentences from given context.

The context relevance score is then computed as:

CR = number of extracted sentencestotal number of sentences in c(q) (2)

## 4 The WikiEval Dataset

To evaluate the proposed framework, we ideally need examples of question-context-answer triples which are annotated with human judgments. We can then verify to what extent our metrics agree with human assessments of faithfulness, answer relevance and context relevance. Since we are not aware of any publicly available datasets that could be used for this purpose, we created a new dataset, which we refer to as WikiEval4. To construct the dataset, we first selected 50 Wikipedia pages covering events that have happened since the start of 20225. In selecting these pages, we prioritised those with recent edits. For each of the 50 pages, we then asked ChatGPT to suggest a question that can be answered based on the introductory section of the page, using the following prompt:

Your task is to formulate a question from given context satisfying the rules given below:

1. The question should be fully answered from the given context.

4https://huggingface.co/datasets/ explodinggradients/WikiEval.

5That is, beyond the reported training cutoff of the model we used in our experiments.

2. The question should be framed from a part that contains non-trivial information.

3. The answer should not contain any links.

4. The question should be of moderate difficulty.

5. The question must be reasonable and must be understood and responded to by humans.

6. Do not use phrases that ‘provided context’, etc in the question context:

We also used ChatGPT to answer the generated question, when given the corresponding introductory section as context, using the following prompt:

Answer the question using the information from the given context.

question: [question] context: [context] All questions were annotated along the three considered quality dimensions by two annotators. Both annotators were fluent in English and were given clear instructions about the meaning of the three considered quality dimensions. For faithfulness and context relevance, the two annotators agreed in around 95% of cases. For answer relevance, they agreed in around 90% of the cases. Disagreements were resolved after a discussion between the annotators. Tables 1, 2 and 3 show examples from the WikiEval dataset, focusing in particular on answers with high and low faithfulness (Table 1), high and low answer relevance (Table 2), and high and low context relevance (Table 3).

Faithfulness To obtain human judgements about faithfulness, we first used ChatGPT to answer the question without access to any additional context.

We then asked the annotators to judge which of the two answers was the most faithful (i.e. the standard one or the one generated without context), given the question and corresponding Wikipedia page.

Answer relevance We first used ChatGPT to obtain candidate answers with lower answer relevance, using the following prompt:

Answer the given question in an incomplete manner.

question: [question] 153 Question Context Answer Who directed the film Oppenheimer and who stars as J. Robert Oppenheimer in the film?

Oppenheimer is a 2023 biographical thriller film written and directed by Christopher Nolan. Based on the 2005 biography American Prometheus by Kai Bird and Martin J. Sherwin, the film chronicles the life of J. Robert Oppenheimer, a theoretical physicist who was pivotal in developing the first nuclear weapons as part of the Manhattan Project, and thereby ushering in the Atomic Age.

Cillian Murphy stars as Oppenheimer, with Emily Blunt as Oppenheimer’s wife Katherine "Kitty" Oppenheimer.

High Faithfulness : Christopher Nolan directed the film Oppenheimer. Cillian Murphy stars as J.

Robert Oppenheimer in the film.

Low Faithfulness : James Cameron directed the film Oppenheimer. Tom Cruise stars as J.

Robert Oppenheimer in the film.

Table 1: Example from WikiEval, showing answers with high and low faithfulness. Question Answer When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?

High answer relevance: The PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India.

Low answer relevance: The scheduled launch date and time for the PSLV-C56 mission have not been provided.The PSLV-C56 mission is an important space mission for India. It aims to launch a satellite into orbit to study weather patterns.

Table 2: Example from WikiEval, showing answers with high and low answer relevance. We then asked human annotators to compare this answer, and indicate which of the two answers had the highest answer relevance.

Context relevance To measure this aspect, we first added additional sentences to the context by scraping back-links to the corresponding Wikipedia page. In this way, we were able to add information to the context that was related but less relevant for answering the question. For the few pages without any back-links, we instead used ChatGPT to complete the given context.

## 5 Experiments

Table 4 analyses the agreement between the metrics proposed in Section 3 and the human assessments from the proposed WikiEval dataset. Each WikiEval instance requires the model to compare two answers or two context fragments. We count how often the answer/context preferred by the model (i.e. with highest estimated faithfulness, answer relevance, or context relevance) coincides with the answer/context preferred by the human annotators. We report the results in terms of accuracy (i.e. the fraction of instances on which the model agrees with the annotators).

To put the results in context, we compare our proposed metrics (shown as RAGA S in Table 4) with two baseline methods. For the first method, shown as GPT Score, we ask ChatGPT to assign a score between 0 and 10 for the three quality dimensions. To this end, we use a prompt that describes the meaning of the quality metric and then asks to score the given answer/context in line with that definition. For instance, for evaluating faithfulness, we used the following prompt:

Faithfulness measures the information consistency of the answer against the given context. Any claims that are made in the answer that cannot be deduced from context should be penalized.

Given an answer and context, assign a score for faithfulness in the range 0-10.

context: [context] answer: [answer] Ties, where the same score is assigned by the LLM to both answer candidates, were broken randomly.

The second baseline, shown as GPT Ranking, instead asks ChatGPT to select the preferred answer/- context. In this case, the prompt again includes a definition of the considered quality metric. For instance, to evaluate answer relevance, we used the following prompt:

Answer Relevancy measures the degree to which a response directly addresses and is appropriate for a given question.

It penalizes the present of redundant information or incomplete answers given a question. Given an question and answer, rank each answer based on Answer Relevancy.

question: [question] 154 Question Context When was the Chimnabai Clock Tower completed, and who was it named after?

High context relevance: The Chimnabai Clock Tower, also known as the Raopura Tower, is a clock tower situated in the Raopura area of Vadodara, Gujarat, India. It was completed in 1896 and named in memory of Chimnabai I (1864–1885), a queen and the first wife of Sayajirao Gaekwad III of Baroda State.

Low context relevance: The Chimnabai Clock Tower, also known as the Raopura Tower, is a clock tower situated in the Raopura area of Vadodara, Gujarat, India. It was completed in 1896 and named in memory of Chimnabai I (1864–1885), a queen and the first wife of Sayajirao Gaekwad III of Baroda State. It was built in Indo-Saracenic architecture style. History. Chimnabai Clock Tower was built in 1896. The tower was named after Chimnabai I (1864–1885), a queen and the first wife of Sayajirao Gaekwad III of Baroda State. It was inaugurated by Mir Kamaluddin Hussainkhan, the last Nawab of Baroda. During the rule of Gaekwad, it was a stoppage for horse drawn trams. The clock tower was erected at the cost of 25,000 (equivalent to 9.2 million or USD 120,000 in 2023). Table 3: Example from WikiEval, showing answers with high and low context relevance. Faith. Ans. Rel. Cont. Rel.

RAGA S 0.95 0.78 0.70 GPT Score 0.72 0.52 0.63 GPT Ranking 0.54 0.40 0.52 Table 4: Agreement with human annotators in pairwise comparisons of faithfulness, answer relevance and context relevance, using the WikEval dataset (accuracy).

answer 1: [answer 1] answer 2: [answer 2] The results in Table 4 show that our proposed metrics are much closer aligned with the human judgements than the predictions from the two baselines. For faithfulness, the RAGA S prediction are in general highly accurate. For answer relevance, the agreement is lower, but this is largely due to the fact that the differences between the two candidate answers are often very subtle. We found context relevance to be the hardest quality dimension to evaluate. In particular, we observed that ChatGPT often struggles with the task of selecting the sentences from the context that are crucial, especially for longer contexts.

## 5.1 Reproducibility

Obtaining reproducible results with (large) language models is challenging. For this reason, reliable software that uses prompts should account not only for hallucinations, but for the fact that several runs of the same experiment under the same configuration might yield different results, e.g., because of an undocumented change in the underlying API, or because of the inherent randomness in neural networks. Furthermore, we require the LLM to generate outputs in structured JSON format. We found that this largely makes RAGA S compatible with different LLMs, and ultimately lowers the error rate when consuming LLM generated text. To measure the effectiveness of the JSON-formatting, we measured the correlation between RAGA S scores in successive runs with and without JSON formatting.

As shown in Figure 1, the scores are clearly more consistent when JSON formatted outputs are used.

Figure 1: We compare the consistency of RAGA S scores across two different runs of the model, with JSON formatting (left) and without (right). The use of JSON formatting leads to more consistent scores.

## 6 Python API

RAGA S provides access to metrics and datasets via an easy-to-use Python API. Its syntax is similar to other well-known libraries such as transformers or datasets. As an example, once installed, loading a dataset, evaluating a pipeline with the desired metrics, and exporting the results to a pandas dataframe can be accomplished with the snippet below. The metrics available at ragas.metrics use OpenAI’s API by default, which requires having the appropriate environment variables set up. It is however possible to experiment with other LLMs 155 for evaluation6.

# import required modules from ragas . metrics import ( answer_relevancy , faithfulness , context_relevancy , ) from ragas import evaluate from datasets import load_dataset # loading the eval dataset amnesty_qa = load_dataset ( ' e xp l o di n g gr a d ie n t s / amnesty_qa' , ' english_v2 ' ) # evaluate from ragas import evaluate result = evaluate ( amnesty_qa [ " eval " ] , metrics = [ faithfulness , answer_relevancy , context_relevancy , ] , ) # export results to pandas dataframe df = result . to_pandas ()

## 7 Conclusions

We have highlighted the need for automated reference-free evaluation of RAG systems. In particular, we have argued the need for an evaluation framework that can assess faithfulness (i.e. is the answer grounded in the retrieved context), answer relevance (i.e. does the answer address the question) and context relevance (i.e. is the retrieved context sufficiently focused). To support the development of such a framework, we have introduced WikiEval, a dataset which human judgements of these three different aspects. Finally, we have also described RAGA S, our implementation of the three considered quality aspects. This framework is easy to use and can provide developers of RAG systems with valuable insights, even in the absence of any ground truth. Our evaluation on WikiEval has shown that the predictions from RAGA S are closely aligned with human judgments, especially for faithfulness and answer relevance.

## 8 Limitations

This paper introduces a toolkit aimed at providing an end-to-end evaluation framework for RAG 6See https://github.com/explodinggradients/ ragas/blob/main/docs/howtos/customisations/llms.

ipynb.

systems. It relies heavily on the performance of the LLMs used for evaluating the different components. While the current set of experiments demonstrate high correlation of these metrics with human judgements, we acknowledge that relying on LLMs comes with known limitations. Therefore, careful reviewing of the suitability of LLMs (ideally prioritizing open models over full-fledged products behind paid APIs) is critical for RAGA S, and other contributions in this space, to nurture a healthy environment. We are also aware of the potential implications of enabling large user bases into using more accurate RAG systems, and therefore we will continue to encourage applications of fair systems on top of RAGA S.
