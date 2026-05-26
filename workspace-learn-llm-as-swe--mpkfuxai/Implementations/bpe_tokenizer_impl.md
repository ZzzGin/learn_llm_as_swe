# BPE Tokenizer Implementation

## BPE Tokenizer Training

### 1. Count Pretokens

As described in the [Tokenizer in LLM Architecture](https://ohmy.md/workspace/mpkfuxai/LLM%20Architecture/1-tokenizer.md#pre-tokenization-regex), pretokens are separated corpus parts that we don't want the BPE merging algorithm to happen across the part  boundaries. In English, pretokens are "mostly" words.

This separation is done by Regex. Here is an example Regex expression used by GPT-4:

```
r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

This step is to count the unique pretokens so that we can do following merges on this dictionaly instead of the raw corpus. This step can be done in parallel by splitting the corpus in chunks - count the pretokens on each chunk - merge the counters.

To split the corpus file safely without cutting words or special tokens (like `<|endoftext|>`) in half, we can use an algorithm that aligns the chunk boundaries to the occurrences of the special document separator token.

Below is an implementation of such a chunking function:

```
def _find_chunk_boundaries(
    input_path: str | Path, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    with open(input_path, "rb") as f:

        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        chunk_size = file_size // desired_num_chunks

        chunk_boundaries = [i * chunk_size 
                 for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            f.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = f.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size
        return sorted(set(chunk_boundaries))
```

### How the Chunking Algorithm Works (With an Example)

Imagine we have a text corpus of **100,000 bytes** stored on disk, and we wish to split it into **4 chunks** (`desired_num_chunks = 4`) for parallel counting. The documents in the file are separated by the special token `b"<|endoftext|>"` (which is 13 bytes long).

Here is a step-by-step trace of how the boundaries are calculated and adjusted:

#### Step 1: Initialize Naive Boundaries

The algorithm first computes a naive chunk size:

$$
\text{chunk\_size} = 100,000 \text{ bytes} / 4 = 25,000 \text{ bytes}
$$

It sets up an initial list of linear byte boundaries spanning the file size: `chunk_boundaries = [0, 25000, 50000, 75000, 100000]`

If we were to cut the file exactly at these offsets, we would likely split words or even break the `<|endoftext|>` token in half, causing issues during tokenization.

#### Step 2: Adjust Intermediate Boundaries

The algorithm iterates through each of the intermediate boundaries (`25000`, `50000`, and `75000`) and adjusts them forward to the nearest occurrence of the special split token (`b"<|endoftext|>"`).

* **Boundary 1 (Initial Guess: `25000`)**:

  * The file pointer seeks to offset `25000` and reads a `mini_chunk` of 4,096 bytes (covering offsets `25000` to `29096`).

  * Suppose the nearest document-separator token `b"<|endoftext|>"` is located at offset `26500` in the file.

  * The algorithm finds this token at index `1500` within the mini-chunk ($26500 - 25000 = 1500$).

  * The boundary is adjusted forward to the exact start of this token: **`26500`**.

* **Boundary 2 (Initial Guess: `50000`)**:

  * The file pointer seeks to offset `50000`.

  * Suppose the nearest `b"<|endoftext|>` is actually at offset `55000`.

  * The first read (`50000` to `54096`) does not find the token. The search pointer (`initial_position`) advances by 4,096 bytes to `54096`.

  * The second read (`54096` to `58192`) contains the token at index `904` of the mini-chunk ($55000 - 54096 = 904$).

  * The boundary is adjusted forward: $54096 + 904 =$ **`55000`**.

* **Boundary 3 (Initial Guess: `75000`)**:

  * The file pointer seeks to offset `75000`.

  * Suppose there are no more `<|endoftext|>` separator tokens between offset `75000` and the end of the file.

  * The loop keeps reading 4,096-byte mini-chunks until it hits the End-of-File (EOF), returning an empty byte string `b""`.

  * The boundary is adjusted to the end of the file: **`100000`**.

#### Step 3: Deduplicate and Sort

After adjusting all intermediate boundaries, the temporary boundary array is: `[0, 26500, 55000, 100000, 100000]`

By running `sorted(set(chunk_boundaries))`, duplicate boundaries are removed, resulting in: `[0, 26500, 55000, 100000]`

#### Final Output Chunks

Even though we requested 4 chunks, the algorithm safely returns 3 chunks to prevent overlaps:

1. **Chunk 1**: bytes `[0, 26500)`

2. **Chunk 2**: bytes `[26500, 55000)`

3. **Chunk 3**: bytes `[55000, 100000)`

Each chunk (except the very first one starting at `0`) begins precisely on a `b"<|endoftext|>"` boundary, allowing each thread to process its chunk as a set of complete, unbroken documents.

### 2. Merge pairs

#### Key Components of the Merging Pipeline

#### 1. Counting Pairs (`_pair_counts`)

To determine which tokens to merge, we must first analyze the adjacent pairs within each individual word (pre-token). The `_pair_counts` function takes a word represented as a tuple of token IDs and returns a dictionary of local pair frequencies.

```
def _pair_counts(word: tuple[int, ...]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for left, right in zip(word, word[1:]):
        counts[(left, right)] += 1
    return counts
```

For example, given the word `"banana"` represented as byte IDs `(98, 97, 110, 97, 110, 97)`:

* It generates the adjacent pairs: `(98, 97)`, `(97, 110)`, `(110, 97)`, `(97, 110)`, and `(110, 97)`.

* The returned counts are: `{(98, 97): 1, (97, 110): 2, (110, 97): 2}`.

#### 2. Selecting the Best Pair (`_select_best_pair`)

Once global pair counts are accumulated, we find the overall most frequent pair using `_select_best_pair`. If there is a tie in frequencies, the tie is broken lexicographically using the underlying byte values of the merged tokens to ensure deterministic behavior across platforms.

```
def _select_best_pair(
    pair_counts: Counter[tuple[int, int]], token_bytes: list[bytes]
) -> tuple[int, int] | None:
    best_pair: tuple[int, int] | None = None
    best_count = 0
    best_bytes: tuple[bytes, bytes] | None = None

    for pair, count in pair_counts.items():
        if count <= 0:
            continue
        pair_bytes = (token_bytes[pair[0]], token_bytes[pair[1]])
        if count > best_count or (
            count == best_count and (best_bytes is None or pair_bytes > best_bytes)
        ):
            best_pair = pair
            best_count = count
            best_bytes = pair_bytes

    return best_pair
```

#### 3. Replacing Pairs inside Sequences (`_merge_pair_in_word`)

When a pair is selected to form a new token, we must substitute all occurrences of that pair within the token lists of our corpus. The substitution is performed sequentially from left to right:

```
def _merge_pair_in_word(
    word: tuple[int, ...], pair: tuple[int, int], new_token_id: int
) -> tuple[int, ...]:
    merged: list[int] = []
    index = 0
    while index < len(word):
        if (
            index + 1 < len(word)
            and word[index] == pair[0]
            and word[index + 1] == pair[1]
        ):
            merged.append(new_token_id)
            index += 2
            continue
        merged.append(word[index])
        index += 1
    return tuple(merged)
```

**Note on Greedy Substitution**: If a sequence has overlapping or repeating pairs like `(1, 1, 1)` and we merge the pair `(1, 1)` into `new_token_id = 99`, the loop processes from left to right. The first two `1`s are merged into `99`, and the remaining `1` is left as is, resulting in `(99, 1)`.

---

### Dynamic Merging Loop in `train_bpe`

Re-scanning the entire corpus to count pairs at each iteration would be incredibly slow ($O(N \cdot M)$ where $N$ is the corpus size and $M$ is the number of merges). To optimize this, `train_bpe` uses an **inverted index** called `pair_to_words`.

#### Data Structures

1. `word_counts`: A dictionary mapping unique pre-token byte sequences to their corpus-wide frequency.

2. `pair_counter`: A global tracker of adjacent pair frequencies, weighted by the parent word frequencies.

3. `pair_to_words`: A mapping of `pair -> {word: occurrences_in_word}`. This allows us to instantly pinpoint which words contain the newly merged pair, ensuring we only update what is necessary.

#### The Update Cycle

For each merge step:

1. Find the best pair to merge using `_select_best_pair`.

2. Map this pair to a new token ID.

3. Identify all words containing this pair via `pair_to_words[best_pair]`.

4. For each affected word:

   * Remove the word's current pair counts from the global `pair_counter`.

   * Remove the word from references in `pair_to_words`.

   * Perform the merge inside the word sequence using `_merge_pair_in_word`.

   * Insert the newly merged sequence into `word_counts`.

   * Calculate the new pair counts for the merged sequence, then add them back into both the global `pair_counter` and the inverted index `pair_to_words`.

---

### Tracing a Merge Step (Example)

Let's walk through a single merge iteration using a small corpus of two words:

* `"hug"` (represented as byte IDs `(104, 117, 103)`) with a frequency of **5**

* `"pug"` (represented as byte IDs `(112, 117, 103)`) with a frequency of **10**

#### Step 1: Initial State

The vocabulary initially contains the 256 base bytes ($0 \text{ to } 255$). The training algorithm populates the initial pairs and indexes.

* **`word_counts`**:

  ```
  {
      (104, 117, 103): 5,  # "hug"
      (112, 117, 103): 10  # "pug"
  }
  ```

* **`pair_counter`**:

  * `(104, 117)` appears 1 time in `"hug"` $\rightarrow 1 \times 5 = 5$

  * `(117, 103)` appears 1 time in `"hug"` and 1 time in `"pug"` $\rightarrow (1 \times 5) + (1 \times 10) = 15$

  * `(112, 117)` appears 1 time in `"pug"` $\rightarrow 1 \times 10 = 10$

  ```
  {
      (117, 103): 15,
      (112, 117): 10,
      (104, 117): 5
  }
  ```

* **`pair_to_words`**:

  ```
  {
      (117, 103): {(104, 117, 103): 1, (112, 117, 103): 1},
      (112, 117): {(112, 117, 103): 1},
      (104, 117): {(104, 117, 103): 1}
  }
  ```

#### Step 2: Selecting and Registering the Merge

The algorithm identifies that `(117, 103)` (representing `'ug'`) is the most frequent pair with a count of **15**.

* A new token ID is generated: `new_token_id = 256`.

* The pair is saved into the list of merges: `merges.append((b'u', b'g'))`.

#### Step 3: Removing Old Pairs for Affected Words

We query `pair_to_words` to find which words contain `(117, 103)`. Both `(104, 117, 103)` and `(112, 117, 103)` are affected. We pop them from `word_counts` and subtract their associated pair counts from the global tracker:

* **For `"hug"` `(104, 117, 103)` (frequency 5)**:

  * Decrement `(104, 117)` by 5 $\rightarrow$ new count: $5 - 5 = 0$ (removed from counter)

  * Decrement `(117, 103)` by 5 $\rightarrow$ new count: $15 - 5 = 10$

* **For `"pug"` `(112, 117, 103)` (frequency 10)**:

  * Decrement `(112, 117)` by 10 $\rightarrow$ new count: $10 - 10 = 0$ (removed from counter)

  * Decrement `(117, 103)` by 10 $\rightarrow$ new count: $10 - 10 = 0$ (removed from counter)

At this exact midpoint, `pair_counter` is temporarily cleared of all original pairs associated with these words.

#### Step 4: Merging and Inserting New Pairs

Next, the algorithm merges the selected pair inside the affected words to produce the updated sequences:

* `(104, 117, 103)` merges into **`(104, 256)`**

* `(112, 117, 103)` merges into **`(112, 256)`**

We insert these merged words back into `word_counts` and calculate their new adjacent pair structures to update `pair_counter` and `pair_to_words`:

* **For `"h" + "ug"` `(104, 256)` (frequency 5)**:

  * Contains the pair `(104, 256)`. We increment `pair_counter[(104, 256)]` by 5.

  * Update `pair_to_words[(104, 256)]` to include `{(104, 256): 1}`.

* **For `"p" + "ug"` `(112, 256)` (frequency 10)**:

  * Contains the pair `(112, 256)`. We increment `pair_counter[(112, 256)]` by 10.

  * Update `pair_to_words[(112, 256)]` to include `{(112, 256): 1}`.

#### Step 5: Final State after Iteration 1

The state is now fully updated and ready for the next iteration:

* **`word_counts`**:

  ```
  {
      (104, 256): 5,
      (112, 256): 10
  }
  ```

* **`pair_counter`**:

  ```
  {
      (112, 256): 10,
      (104, 256): 5
  }
  ```

* **`pair_to_words`**:

  ```
  {
      (112, 256): {(112, 256): 1},
      (104, 256): {(104, 256): 1}
  }
  ```

In the next iteration, the algorithm will select the pair `(112, 256)` (representing `b'pug'`) to merge, repeating this dynamic update cycle until the target vocabulary size is achieved.