# 1 - Tokenizer

LLMs do not process raw text directly. As the first step of the training or inference process, text must be converted into a sequence of numbers (token IDs) that the neural network can compute.

---

## Text Corpus

A very large raw text file seperated by special token `<|endoftext|>`. It looks like:

```
Once upon a time, there was a <...omitted...> their work.
One day, a man wanted <...omitted...> would break.
The bow did not <...omitted...> their best bow.
<|endoftext|>
Once upon a time, there was <...omitted...> of fun together.
One day, Lily's <...omitted...> the cookie and the bone for later.
Lily put the <...omitted...> are fresh and clean.
<|endoftext|>
...
```

## Byte Pair Encoding (BPE) Tokenizer

Byte Pair Encoding (BPE) is a subword tokenization method that iteratively merges the most frequent adjacent pairs of bytes or characters in a text corpus to construct a vocabulary of a specified size.

### Pre-tokenization Regex

To prevent invalid merges across word boundaries or distinct grammatical groups, a **pre-tokenization** step is first applied using a regular expression. This splits the raw text into distinct structural units (pre-tokens) of alphabetic letters, numbers, punctuation, and spaces before the BPE merging process begins.

Our tokenizer uses a GPT-4-compatible regex to segment strings prior to learning or applying token merges:

```
regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    regex.V1,
)
```

Consider the following input string:

```
"Hello world, I'm 10 years old!"
```

Applying the regular expression splits the string into distinct pre-tokens:

1. `"Hello"` (optional space + letters)

2. `" world"` (optional space + letters)

3. `","` (punctuation)

4. `" I"` (optional space + letters)

5. `"'m"` (contraction suffix)

6. `" 10"` (optional space + numbers)

7. `" years"` (optional space + letters)

8. `" old"` (optional space + letters)

9. `"!"` (punctuation)

### BPE Training

The core objective of training a Byte Pair Encoding (BPE) tokenizer is to build a vocabulary of common character sequences and subword units by starting with individual bytes and iteratively merging the most frequent adjacent pairs.

The training process operates as follows:

1. **Initialize the Vocabulary**: The vocabulary is seeded with the 256 raw byte values (0–255). At the start, the text `"Hello world, I'm 10 years old!"` is represented as a sequence of these base byte tokens: `[72, 101, 108, 108, 111, 32, ...]` (representing `'H'`, `'e'`, `'l'`, `'l'`, `'o'`, `' '`, and so on).

2. **Pre-tokenization**: The raw training text is split into smaller, structural pre-tokens using the regex: `["Hello", " world", ",", " I", "'m", " 10", " years", " old", "!"]`. This ensures that merges only occur within these logical boundaries and do not cross them; for example, the `'o'` at the end of `"Hello"` and the space `' '` at the start of `" world"` will never be merged.

3. **Count Adjacent Pairs**: The algorithm counts the frequency of all adjacent token pairs strictly within each pre-token boundary. For the pre-token `"Hello"`, the counted pairs are `('H', 'e')`, `('e', 'l')`, `('l', 'l')`, and `('l', 'o')`.

4. **Select and Merge**: The most frequent adjacent pair across the entire corpus is identified. If `('l', 'l')` has the highest frequency, a new token (ID 256) is created in the vocabulary to represent `'ll'`, and all occurrences are merged. The representation of `"Hello"` is updated from `['H', 'e', 'l', 'l', 'o']` to `['H', 'e', 'll', 'o']`.

5. **Iterate**: This loop of counting adjacent pairs, identifying the most frequent pair, and merging them is repeated. In subsequent iterations, `('e', 'll')` might merge to become `'ell'`, updating `"Hello"` further to `['H', 'ell', 'o']`. With each merge step, the vocabulary size grows by one. The loop terminates once the target vocabulary size is reached.

6. **Append Special Tokens**: Finally, any required special control tokens (such as `<|endoftext|>`) are added to the end of the vocabulary.

### Vocabulary and Merges

The output of the BPE training process consists of two primary data structures:

1. **Merges**: An ordered dictionary mapping a pair of token IDs to a new, merged token ID. The order of these merges is critical, as it defines the precise sequence in which tokens must be combined during encoding.

2. **Vocabulary**: A mapping of token IDs to their corresponding byte-string representations. This is used during decoding to reconstruct the original text.

To illustrate how these structures are represented and utilized, let's look at a concrete example using a tiny toy corpus containing the words `"hug"`, `"pug"`, and `"pun"`.

#### The Merges Dictionary

During training, the algorithm first identifies individual character bytes. Suppose their ASCII/byte values are:

* `'g'`: `103`

* `'h'`: `104`

* `'n'`: `110`

* `'p'`: `112`

* `'u'`: `117`

As the tokenizer trains, it records each merge in the order it occurs. The resulting merges dictionary might look like this:

```
merges = {
    (117, 103): 256,  # Merges 'u' (117) and 'g' (103) -> 'ug' (256)
    (112, 117): 257,  # Merges 'p' (112) and 'u' (117) -> 'pu' (257)
    (257, 110): 258,  # Merges 'pu' (257) and 'n' (110) -> 'pun' (258)
}
```

The sequence of keys in this dictionary represents the exact order of merge operations. When encoding a new text like `"hug"`, the tokenizer will look at the byte IDs `[104, 117, 103]`. It checks if any adjacent pair exists in the merges dictionary. Since `(117, 103)` exists, it merges them into `256`, resulting in `[104, 256]`. No more merges apply, so the final encoded sequence is `[104, 256]`.

#### The Vocabulary

The vocabulary is built by starting with the 256 base bytes (0 to 255) and iteratively appending the byte representations of the newly merged tokens. Finally, any special tokens (like `<|endoftext|>`) are assigned the next available IDs.

For our toy example, the vocabulary is represented as a dictionary mapping token IDs to Python `bytes`:

```
vocab = {
    # --- Base Vocabulary (IDs 0 - 255) ---
    0: b'\x00',
    1: b'\x01',
    # ...
    103: b'g',
    104: b'h',
    110: b'n',
    112: b'p',
    117: b'u',
    # ...
    255: b'\xff',

    # --- Merged Tokens (IDs 256+) ---
    256: b'ug',
    257: b'pu',
    258: b'pun',

    # --- Special Tokens ---
    259: b'<|endoftext|>'
}
```

### BPE Encoding

To encode a raw string into a sequence of token IDs using a trained BPE tokenizer, we reverse the merge process. However, because BPE merges are hierarchical and order-dependent, we must apply merges strictly in the order they were learned during training (from the lowest merge rank to the highest).

The encoding process consists of the following steps:

1. **Pre-tokenization**: The input string is split into pre-tokens using the GPT-4 regular expression.

2. **Byte Conversion**: Each pre-token is converted into an array of its individual byte values (0–255).

3. **Iterative Merging**: For each pre-token's byte sequence, we look at all adjacent pairs. We identify the pair that appears earliest in our `merges` dictionary (i.e., has the lowest merge rank) and merge it. We repeat this lookup and merge process on the updated sequence until no more merges from the dictionary can be applied.

4. **Concatenation**: The final token IDs of all pre-tokens are concatenated together to form the complete encoded representation of the input string.

Let's trace how the string `"Hello world, I'm 10 years old!"` is encoded.

#### Step 1: Pre-tokenization

Applying the regex splits the input into 9 independent pre-tokens:

```
["Hello", " world", ",", " I", "'m", " 10", " years", " old", "!"]
```

Each of these pre-tokens is encoded individually. Merges will never occur across the boundaries of these pre-tokens.

#### Step 2: Set up a Hypothetical Merges Dictionary

To demonstrate the merging mechanics, let's assume our trained `merges` dictionary contains the following rules (sorted by their priority/rank):

| Pair | Merged Token ID | Resulting Subword | Priority (Rank) |
| --- | --- | --- | --- |
| (108, 108) | 256 | 'll' | 1 (First) |
| (101, 256) | 257 | 'ell' | 2 |
| (72, 257) | 258 | 'Hell' | 3 |
| (119, 111) | 259 | 'wo' | 4 |
| (259, 114) | 260 | 'wor' | 5 |
| (32, 260) | 261 | ' wor' | 6 (Last) |

#### Step 3: Merging within Pre-Tokens

We apply the merges to each pre-token. Let's trace the merging process for the first two pre-tokens.

##### Token 1: `"Hello"`

* **Initial state**: Represented as raw bytes `['H', 'e', 'l', 'l', 'o']` mapped to ASCII values:

  ```
  [72, 101, 108, 108, 111]
  ```

* **Iteration 1**:

  * Current adjacent pairs: `(72, 101)`, `(101, 108)`, `(108, 108)`, `(108, 111)`.

  * Out of these, only `(108, 108)` exists in our merges dictionary (with rank 1).

  * We merge them into `256`. The sequence becomes:

    ```
    [72, 101, 256, 111]  # ('H', 'e', 'll', 'o')
    ```

* **Iteration 2**:

  * Current adjacent pairs: `(72, 101)`, `(101, 256)`, `(256, 111)`.

  * Only `(101, 256)` exists in our merges dictionary (with rank 2).

  * We merge them into `257`. The sequence becomes:

    ```
    [72, 257, 111]       # ('H', 'ell', 'o')
    ```

* **Iteration 3**:

  * Current adjacent pairs: `(72, 257)`, `(257, 111)`.

  * Only `(72, 257)` exists in our merges dictionary (with rank 3).

  * We merge them into `258`. The sequence becomes:

    ```
    [258, 111]            # ('Hell', 'o')
    ```

* **Iteration 4**:

  * Current adjacent pairs: `(258, 111)`.

  * This pair does not exist in our merges dictionary. The process stops.

  * **Final token sequence for `"Hello"`**: `[258, 111]`

##### Token 2: `" world"`

* **Initial state**: Represented as raw bytes `[' ', 'w', 'o', 'r', 'l', 'd']` mapped to ASCII values:

  ```
  [32, 119, 111, 114, 108, 100]
  ```

* **Iteration 1**:

  * Adjacent pairs: `(32, 119)`, `(119, 111)`, `(111, 114)`, `(114, 108)`, `(108, 100)`.

  * The only matching pair in our merges dictionary is `(119, 111)` (rank 4).

  * We merge them into `259`. The sequence becomes:

    ```
    [32, 259, 114, 108, 100]  # (' ', 'wo', 'r', 'l', 'd')
    ```

* **Iteration 2**:

  * Adjacent pairs: `(32, 259)`, `(259, 114)`, `(114, 108)`, `(108, 100)`.

  * The only matching pair is `(259, 114)` (rank 5).

  * We merge them into `260`. The sequence becomes:

    ```
    [32, 260, 108, 100]       # (' ', 'wor', 'l', 'd')
    ```

* **Iteration 3**:

  * Adjacent pairs: `(32, 260)`, `(260, 108)`, `(108, 100)`.

  * The only matching pair is `(32, 260)` (rank 6).

  * We merge them into `261`. The sequence becomes:

    ```
    [261, 108, 100]            # (' wor', 'l', 'd')
    ```

* **Iteration 4**:

  * Adjacent pairs: `(261, 108)`, `(108, 100)`.

  * None of these pairs exist in our merges dictionary. The process stops.

  * **Final token sequence for `" world"`**: `[261, 108, 100]`

##### Remaining Tokens

For simplicity, let's assume the remaining pre-tokens do not have any matching rules in our merges dictionary. They are simply converted directly to their individual byte sequences:

* `","` $\rightarrow$ `[44]`

* `" I"` $\rightarrow$ `[32, 73]`

* `"'m"` $\rightarrow$ `[39, 109]`

* `" 10"` $\rightarrow$ `[32, 49, 48]`

* `" years"` $\rightarrow$ `[32, 121, 101, 97, 114, 115]`

* `" old"` $\rightarrow$ `[32, 111, 108, 100]`

* `"!"` $\rightarrow$ `[33]`

#### Step 4: Concatenation

Finally, we concatenate the sequences from all pre-tokens to yield the fully encoded array of token IDs:

```
[258, 111, 261, 108, 100, 44, 32, 73, 39, 109, 32, 49, 48, 32, 121, 101, 97, 114, 115, 32, 111, 108, 100, 33]
```

By looking up these IDs in the `vocab` dictionary, we can map them back to their exact byte-string representations whenever we need to decode them.

### BPE Decoding

Decoding a sequence of token IDs back into a raw string is a straightforward and computationally efficient process. Unlike encoding, which requires evaluating merge rules sequentially, decoding only requires looking up the token IDs in the pre-built `vocab` dictionary and concatenating their corresponding byte sequences.

The decoding process consists of three main steps:

1. **Vocabulary Lookup**: Map each token ID in the sequence to its corresponding byte-string representation using the `vocab` dictionary.

2. **Byte Concatenation**: Concatenate all mapped byte strings in order into a single, continuous byte sequence.

3. **UTF-8 Decoding**: Decode the final byte sequence back into a standard Unicode string (typically using UTF-8, with a fallback like `errors="replace"` to handle any incomplete or invalid byte sequences safely).

Let's trace this process using the exact token ID sequence produced in the previous encoding example:

```
[258, 111, 261, 108, 100, 44, 32, 73, 39, 109, 32, 49, 48, 32, 121, 101, 97, 114, 115, 32, 111, 108, 100, 33]
```

#### Step 1: Vocabulary Lookup

We map each token ID back to its byte-string representation using the `vocab` dictionary:

| Token ID | Decoded Byte Value | Decoded Subword Text |
| --- | --- | --- |
| 258 | b'Hell' | "Hell" |
| 111 | b'o' | "o" |
| 261 | b' wor' | " wor" |
| 108 | b'l' | "l" |
| 100 | b'd' | "d" |
| 44 | b',' | "," |
| 32 | b' ' | " " |
| 73 | b'I' | "I" |
| 39 | b"'" | "'" |
| 109 | b'm' | "m" |
| 32 | b' ' | " " |
| 49 | b'1' | "1" |
| 48 | b'0' | "0" |
| 32 | b' ' | " " |
| 121 | b'y' | "y" |
| 101 | b'e' | "e" |
| 97 | b'a' | "a" |
| 114 | b'r' | "r" |
| 115 | b's' | "s" |
| 32 | b' ' | " " |
| 111 | b'o' | "o" |
| 108 | b'l' | "l" |
| 100 | b'd' | "d" |
| 33 | b'!' | "!" |

#### Step 2: Byte Concatenation

Next, we concatenate all the mapped byte strings together in their exact sequence:

```
concatenated_bytes = (
    b'Hell' + b'o' + b' wor' + b'l' + b'd' + b',' + b' ' + b'I' + b"'" + b'm' +
    b' ' + b'1' + b'0' + b' ' + b'y' + b'e' + b'a' + b'r' + b's' + b' ' + b'o' +
    b'l' + b'd' + b'!'
)
```

This yields the unified byte array:

```
b"Hello world, I'm 10 years old!"
```

#### Step 3: UTF-8 Decoding

Finally, we decode the byte array back into a human-readable Python string:

```
decoded_string = concatenated_bytes.decode('utf-8', errors='replace')
```

Which perfectly recovers our original input string:

```
"Hello world, I'm 10 years old!"
```

Because BPE tokenizers operate natively on raw bytes (guaranteeing that any of the 256 individual bytes can always be matched back to a base token), the decoding process is exceptionally robust. Even if the model generates a malformed sequence of token IDs, the decoder will seamlessly assemble the underlying bytes and reconstruct them into valid Unicode (or replace invalid characters) without throwing an error.