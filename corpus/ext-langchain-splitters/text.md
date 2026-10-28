# LangChain text splitters docs

# src/oss/integrations/splitters/index.mdx

:::python
    ```bash pip
    pip install -U langchain-text-splitters
    ```

    ```bash uv
    uv add langchain-text-splitters
    ```
:::

:::js

    ```bash npm
    npm install @langchain/textsplitters @langchain/core
    # Requires Node.js 22+
    ```

    ```bash pnpm
    pnpm add @langchain/textsplitters @langchain/core
    # Requires Node.js 22+
    ```

    ```bash yarn
    yarn add @langchain/textsplitters @langchain/core
    # Requires Node.js 22+
    ```

    ```bash bun
    bun add @langchain/textsplitters @langchain/core
    # Requires Node.js 22+
    ```
:::

**Text splitters** break large docs into smaller chunks that will be retrievable individually and fit within model context window limit.

There are several strategies for splitting documents, each with its own advantages.

:::python
:::

    For most use cases, start with the [`RecursiveCharacterTextSplitter`](/oss/integrations/splitters/recursive_text_splitter). It provides a solid balance between keeping context intact and managing chunk size. This default strategy works well out of the box, and you should only consider adjusting it if you need to fine-tune performance for your specific application.

### Text structure-based

Text is naturally organized into hierarchical units such as paragraphs, sentences, and words. We can leverage this inherent structure to inform our splitting strategy, creating split that maintain natural language flow, maintain semantic coherence within split, and adapts to varying levels of text granularity. LangChain's `RecursiveCharacterTextSplitter` implements this concept:

- The [`RecursiveCharacterTextSplitter`](/oss/integrations/splitters/recursive_text_splitter) attempts to keep larger units (e.g., paragraphs) intact.
- If a unit exceeds the chunk size, it moves to the next level (e.g., sentences).
- This process continues down to the word level if necessary.

Example usage:

:::python
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_text(document)
```
:::
:::js
```ts

const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 100, chunkOverlap: 0 })
const texts = splitter.splitText(document)
```
:::

**Available text splitters**:
- [Recursively split text](/oss/integrations/splitters/recursive_text_splitter)

### Length-based

An intuitive strategy is to split documents based on their length. This simple yet effective approach ensures that each chunk doesn't exceed a specified size limit. Key benefits of length-based splitting:

- Straightforward implementation
- Consistent chunk sizes
- Easily adaptable to different model requirements

Types of length-based splitting:

- Token-based: Splits text based on the number of tokens, which is useful when working with language models.
- Character-based: Splits text based on the number of characters, which can be more consistent across different types of text.

Example implementation using LangChain's `CharacterTextSplitter` with token-based splitting:

:::python
```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(document)
```
:::
:::js
```ts

const splitter = new TokenTextSplitter({ encodingName: "cl100k_base", chunkSize: 100, chunkOverlap: 0 })
const texts = splitter.splitText(document)
```
:::

**Available text splitters**:
- [Split by tokens](/oss/integrations/splitters/split_by_token)
- [Split by characters](/oss/integrations/splitters/character_text_splitter)

### Document structure-based

Some documents have an inherent structure, such as HTML, Markdown, or JSON files. In these cases, it's beneficial to split the document based on its structure, as it often naturally groups semantically related text. Key benefits of structure-based splitting:

- Preserves the logical organization of the document
- Maintains context within each chunk
- Can be more effective for downstream tasks like retrieval or summarization

:::python
Examples of structure-based splitting:

- Markdown: Split based on headers (e.g., `#`, `##`, `###`)
- HTML: Split using tags
- JSON: Split by object or array elements
- Code: Split by functions, classes, or logical blocks

**Available text splitters**:
- [Split Markdown](/oss/integrations/splitters/markdown_header_metadata_splitter)
- [Split JSON](/oss/integrations/splitters/recursive_json_splitter)
- [Split code](/oss/integrations/splitters/code_splitter)
- [Split HTML](/oss/integrations/splitters/split_html)
:::
:::js
**Available text splitters**:
- [Split code](/oss/integrations/splitters/code_splitter)
:::

# src/oss/integrations/splitters/recursive_text_splitter.mdx

This [text splitter](/oss/integrations/splitters/) is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is `["\n\n", "\n", " ", ""]`. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

1. How the text is split: by list of characters.
2. How the chunk size is measured: by number of characters.

Below we show example usage.

:::python
```shell
pip install -qU langchain-text-splitters
```
:::
:::js
```bash npm
npm install @langchain/textsplitters
```

```bash pnpm
pnpm install @langchain/textsplitters
```

```bash yarn
yarn add @langchain/textsplitters
```

```bash bun
bun add @langchain/textsplitters
```
:::

:::python
To obtain the string content directly, use `.split_text`.

To create LangChain @[Document] objects (e.g., for use in downstream tasks), use `.create_documents`.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load example document
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
print(texts[1])
```
```python
page_content='Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and'
page_content='of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.'
```

```python
print(text_splitter.split_text(state_of_the_union)[:2])
```

```python
['Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and',
 'of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.']
```
:::

:::js
To obtain the string content directly, use `.splitText`.

To create LangChain @[Document] objects (e.g., for use in downstream tasks), use `.createDocuments`.

```ts

const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 100, chunkOverlap: 0 })
const texts = splitter.createDocuments([{ pageContent: "..." }])
```
```javascript
[
  { pageContent: "...", metadata: {} },
]
```
:::

Let's go through the parameters set above for `RecursiveCharacterTextSplitter`:

:::python
- `chunk_size`: The maximum size of a chunk, where size is determined by the `length_function`.
- `chunk_overlap`: Target overlap between chunks. Overlapping chunks helps to mitigate loss of information when context is divided between chunks.
- `length_function`: Function determining the chunk size.
- `is_separator_regex`: Whether the separator list (defaulting to `["\n\n", "\n", " ", ""]`) should be interpreted as regex.
:::

:::js
- `chunkSize`: The maximum size of a chunk, where size is determined by the `lengthFunction`.
- `chunkOverlap`: Target overlap between chunks. Overlapping chunks helps to mitigate loss of information when context is divided between chunks.
:::

### Splitting text from languages without word boundaries

Some writing systems do not have [word boundaries](https://en.wikipedia.org/wiki/Category:Writing_systems_without_word_boundaries), for example Chinese, Japanese, and Thai. Splitting text with the default separator list of `["\n\n", "\n", " ", ""]` can cause words to be split between chunks. To keep words together, you can override the list of separators to include additional punctuation:

* Add ASCII full-stop "`.`", [Unicode fullwidth](https://en.wikipedia.org/wiki/Halfwidth_and_Fullwidth_Forms_(Unicode_block)) full stop "`．`" (used in Chinese text), and [ideographic full stop](https://en.wikipedia.org/wiki/CJK_Symbols_and_Punctuation) "`。`" (used in Japanese and Chinese)
* Add [Zero-width space](https://en.wikipedia.org/wiki/Zero-width_space) used in Thai, Myanmar, Kmer, and Japanese.
* Add ASCII comma "`,`", Unicode fullwidth comma "`，`", and Unicode ideographic comma "`、`"

:::python
```python
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    # Existing args
)
```
:::
:::js
```ts
const splitter = new RecursiveCharacterTextSplitter({
  separators: [
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "\u200b",  // Zero-width space
    "\uff0c",  // Fullwidth comma
    "\u3001",  // Ideographic comma
    "\uff0e",  // Fullwidth full stop
    "\u3002",  // Ideographic stop
    "",
  ],
});
```
:::

# src/oss/integrations/splitters/split_by_token.mdx

Language models have a token limit. You should not exceed the token limit. When you [split your text](/oss/integrations/splitters/) into chunks it is therefore a good idea to count the number of tokens. There are many tokenizers. When you count tokens in your text you should use the same tokenizer as used in the language model.

:::python
### tiktoken

**[tiktoken](https://github.com/openai/tiktoken) is a fast `BPE` tokenizer created by `OpenAI`.**


We can use `tiktoken` to estimate tokens used. It will probably be more accurate for the OpenAI models.

1. How the text is split: by character passed in.
2. How the chunk size is measured: by `tiktoken` tokenizer.

@[CharacterTextSplitter], @[RecursiveCharacterTextSplitter], and @[TokenTextSplitter] can be used with `tiktoken` directly.


```python
pip install --upgrade --quiet langchain-text-splitters tiktoken
```


```python
from langchain_text_splitters import CharacterTextSplitter

# This is a long document we can split up.
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()
```

To split with a @[CharacterTextSplitter] and then merge chunks with `tiktoken`, use its `.from_tiktoken_encoder()` method. Note that splits from this method can be larger than the chunk size measured by the `tiktoken` tokenizer.

The `.from_tiktoken_encoder()` method takes either `encoding_name` as an argument (e.g. `cl100k_base`), or the `model_name` (e.g. `gpt-4`). All additional arguments like `chunk_size`, `chunk_overlap`, and `separators` are used to instantiate `CharacterTextSplitter`:


```python
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(state_of_the_union)
```


```python
print(texts[0])
```
```text
Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.

Last year COVID-19 kept us apart. This year we are finally together again.

Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans.

With a duty to one another to the American people to the Constitution.
```
To implement a hard constraint on the chunk size, we can use `RecursiveCharacterTextSplitter.from_tiktoken_encoder`, where each split will be recursively split if it has a larger size:


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=100,
    chunk_overlap=0,
)
```

We can also load a `TokenTextSplitter` splitter, which works with `tiktoken` directly and will ensure each split is smaller than chunk size.


```python
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

texts = text_splitter.split_text(state_of_the_union)
print(texts[0])
```
```text
Madam Speaker, Madam Vice President, our
```
Some written languages (e.g. Chinese and Japanese) have characters which encode to two or more tokens. Using the `TokenTextSplitter` directly can split the tokens for a character between two chunks causing malformed Unicode characters. Use `RecursiveCharacterTextSplitter.from_tiktoken_encoder` or `CharacterTextSplitter.from_tiktoken_encoder` to ensure chunks contain valid Unicode strings.
:::

:::js
### js-tiktoken

**[js-tiktoken](https://github.com/dqbd/tiktoken) is a JavaScript vesrion of the `BPE` tokenizer created by `OpenAI`.**

We can use `tiktoken` to estimate tokens used using @[TokenTextSplitter]. It will probably be more accurate for OpenAI models.

1. How the text is split: by character passed in.
2. How the chunk size is measured: by `tiktoken` tokenizer.

```bash npm
npm install @langchain/textsplitters
```

```bash pnpm
pnpm install @langchain/textsplitters
```

```bash yarn
yarn add @langchain/textsplitters
```

```bash bun
bun add @langchain/textsplitters
```

```ts

// Example: read a long document
const stateOfTheUnion = readFileSync("state_of_the_union.txt", "utf8");
```

To split with a @[TokenTextSplitter] and then merge chunks with `tiktoken`, pass in an `encodingName` (e.g. cl100k_base) when initializing the @[TokenTextSplitter]. Note that splits from this method can be larger than the chunk size measured by the `tiktoken` tokenizer.

```ts

// Example: use cl100k_base encoding
const splitter = new TokenTextSplitter({ encodingName: "cl100k_base", chunkSize: 10, chunkOverlap: 0 });

const texts = splitter.splitText(stateOfTheUnion);
console.log(texts[0]);
```
```text
Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.

Last year COVID-19 kept us apart. This year we are finally together again.

Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans.

With a duty to one another to the American people to the Constitution.
```
:::

:::python
### spaCy

[spaCy](https://spacy.io/) is an open-source software library for advanced natural language processing, written in the programming languages Python and Cython.

LangChain implements splitters based on the [spaCy tokenizer](https://spacy.io/api/tokenizer).

1. How the text is split: by `spaCy` tokenizer.
2. How the chunk size is measured: by number of characters.


```python
pip install --upgrade --quiet  spacy
```


```python
# This is a long document we can split up.
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()
```


```python
from langchain_text_splitters import SpacyTextSplitter

text_splitter = SpacyTextSplitter(chunk_size=1000)

texts = text_splitter.split_text(state_of_the_union)
print(texts[0])
```
```text
Madam Speaker, Madam Vice President, our First Lady and Second Gentleman.

Members of Congress and the Cabinet.

Justices of the Supreme Court.

My fellow Americans.



Last year COVID-19 kept us apart.

This year we are finally together again.



Tonight, we meet as Democrats Republicans and Independents.

But most importantly as Americans.



With a duty to one another to the American people to the Constitution.



And with an unwavering resolve that freedom will always triumph over tyranny.



Six days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways.

But he badly miscalculated.



He thought he could roll into Ukraine and the world would roll over.

Instead he met a wall of strength he never imagined.



He met the Ukrainian people.



From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world.
```
### SentenceTransformers

The [`SentenceTransformersTokenTextSplitter`](https://reference.langchain.com/python/langchain-text-splitters/sentence_transformers/SentenceTransformersTokenTextSplitter) is a specialized text splitter for use with the sentence-transformer models. The default behaviour is to split the text into chunks that fit the token window of the sentence transformer model that you would like to use.

To split text and constrain token counts according to the sentence-transformers tokenizer, instantiate a `SentenceTransformersTokenTextSplitter`. You can optionally specify:

- `chunk_overlap`: integer count of token overlap;
- `model_name`: sentence-transformer model name, defaulting to `"sentence-transformers/all-mpnet-base-v2"`;
- `tokens_per_chunk`: desired token count per chunk.


```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
text = "Lorem "

count_start_and_stop_tokens = 2
text_token_count = splitter.count_tokens(text=text) - count_start_and_stop_tokens
print(text_token_count)
```
```text
2
```

```python
token_multiplier = splitter.maximum_tokens_per_chunk // text_token_count + 1

# `text_to_split` does not fit in a single chunk
text_to_split = text * token_multiplier

print(f"tokens in text to split: {splitter.count_tokens(text=text_to_split)}")
```
```text
tokens in text to split: 514
```

```python
text_chunks = splitter.split_text(text=text_to_split)

print(text_chunks[1])
```
```text
lorem
```
### NLTK

**[The Natural Language Toolkit](https://en.wikipedia.org/wiki/Natural_Language_Toolkit), or more commonly [NLTK](https://www.nltk.org/), is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language.**


Rather than just splitting on "\n\n", we can use `NLTK` to split based on [NLTK tokenizers](https://www.nltk.org/api/nltk.tokenize.html).

1. How the text is split: by `NLTK` tokenizer.
2. How the chunk size is measured: by number of characters.


```python
# pip install nltk
```


```python
# This is a long document we can split up.
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()
```


```python
from langchain_text_splitters import NLTKTextSplitter

text_splitter = NLTKTextSplitter(chunk_size=1000)
```


```python
texts = text_splitter.split_text(state_of_the_union)
print(texts[0])
```
```text
Madam Speaker, Madam Vice President, our First Lady and Second Gentleman.

Members of Congress and the Cabinet.

Justices of the Supreme Court.

My fellow Americans.

Last year COVID-19 kept us apart.

This year we are finally together again.

Tonight, we meet as Democrats Republicans and Independents.

But most importantly as Americans.

With a duty to one another to the American people to the Constitution.

And with an unwavering resolve that freedom will always triumph over tyranny.

Six days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways.

But he badly miscalculated.

He thought he could roll into Ukraine and the world would roll over.

Instead he met a wall of strength he never imagined.

He met the Ukrainian people.

From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world.

Groups of citizens blocking tanks with their bodies.
```
### KoNLPY

[KoNLPy: Korean NLP in Python](https://konlpy.org/en/latest/) is a Python package for natural language processing (NLP) of the Korean language.

Token splitting involves the segmentation of text into smaller, more manageable units called tokens. These tokens are often words, phrases, symbols, or other meaningful elements crucial for further processing and analysis. In languages like English, token splitting typically involves separating words by spaces and punctuation marks. The effectiveness of token splitting largely depends on the tokenizer's understanding of the language structure, ensuring the generation of meaningful tokens. Since tokenizers designed for the English language are not equipped to understand the unique semantic structures of other languages, such as Korean, they cannot be effectively used for Korean language processing.

#### Token splitting for korean with KoNLPy's kkma analyzer
In case of Korean text, KoNLPY includes at morphological analyzer called `Kkma` (Korean Knowledge Morpheme Analyzer). `Kkma` provides detailed morphological analysis of Korean text. It breaks down sentences into words and words into their respective morphemes, identifying parts of speech for each token. It can segment a block of text into individual sentences, which is particularly useful for processing long texts.

#### Usage considerations
While `Kkma` is renowned for its detailed analysis, it is important to note that this precision may impact processing speed. Thus, `Kkma` is best suited for applications where analytical depth is prioritized over rapid text processing.


```python
# pip install konlpy
```


```python
# This is a long Korean document that we want to split up into its component sentences.
with open("./your_korean_doc.txt") as f:
    korean_document = f.read()
```


```python
from langchain_text_splitters import KonlpyTextSplitter

text_splitter = KonlpyTextSplitter()
```


```python
texts = text_splitter.split_text(korean_document)
# The sentences are split with "\n\n" characters.
print(texts[0])
```
```text
춘향전 옛날에 남원에 이 도령이라는 벼슬아치 아들이 있었다.

그의 외모는 빛나는 달처럼 잘생겼고, 그의 학식과 기예는 남보다 뛰어났다.

한편, 이 마을에는 춘향이라는 절세 가인이 살고 있었다.

춘 향의 아름다움은 꽃과 같아 마을 사람들 로부터 많은 사랑을 받았다.

어느 봄날, 도령은 친구들과 놀러 나갔다가 춘 향을 만 나 첫 눈에 반하고 말았다.

두 사람은 서로 사랑하게 되었고, 이내 비밀스러운 사랑의 맹세를 나누었다.

하지만 좋은 날들은 오래가지 않았다.

도령의 아버지가 다른 곳으로 전근을 가게 되어 도령도 떠나 야만 했다.

이별의 아픔 속에서도, 두 사람은 재회를 기약하며 서로를 믿고 기다리기로 했다.

그러나 새로 부임한 관아의 사또가 춘 향의 아름다움에 욕심을 내 어 그녀에게 강요를 시작했다.

춘 향 은 도령에 대한 자신의 사랑을 지키기 위해, 사또의 요구를 단호히 거절했다.

이에 분노한 사또는 춘 향을 감옥에 가두고 혹독한 형벌을 내렸다.

이야기는 이 도령이 고위 관직에 오른 후, 춘 향을 구해 내는 것으로 끝난다.

두 사람은 오랜 시련 끝에 다시 만나게 되고, 그들의 사랑은 온 세상에 전해 지며 후세에까지 이어진다.

- 춘향전 (The Tale of Chunhyang)
```
### Hugging Face tokenizer

[Hugging Face](https://huggingface.co/docs/tokenizers/index) has many tokenizers.

We use Hugging Face tokenizer, the [GPT2TokenizerFast](https://huggingface.co/Ransaka/gpt2-tokenizer-fast) to count the text length in tokens.

1. How the text is split: by character passed in.
2. How the chunk size is measured: by number of tokens calculated by the `Hugging Face` tokenizer.


```python
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
```


```python
# This is a long document we can split up.
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()
from langchain_text_splitters import CharacterTextSplitter
```


```python
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(state_of_the_union)
```


```python
print(texts[0])
```
```text
Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.

Last year COVID-19 kept us apart. This year we are finally together again.

Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans.

With a duty to one another to the American people to the Constitution.
```
:::
