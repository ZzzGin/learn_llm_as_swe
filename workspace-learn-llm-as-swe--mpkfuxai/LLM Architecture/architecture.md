# Architecture

This document provides a high-level overview of the modern Large Language Model (LLM) architecture detailed in this section. The model is an autoregressive, decoder-only Transformer utilizing modern improvements such as Pre-LN RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU activations, and shared Input-Output embeddings.

### High-Level Architecture Overview

```mermaid
graph TD
    %% Color Palette Configurations
    classDef data fill:#e1f5fe,stroke:#039be5,stroke-width:2px;
    classDef layer fill:#ede7f6,stroke:#5e35b1,stroke-width:2px;
    classDef block fill:#fff3e0,stroke:#f57c00,stroke-width:3px;
    classDef math fill:#f1f8e9,stroke:#558b2f,stroke-width:2px;

    %% Data Inputs
    TextInput([Raw Text Input]) --> Tokenizer[BPE Tokenizer]:::layer
    Tokenizer -->|Token IDs| TokenEmbed[Token Embedding Table]:::layer
    TokenEmbed -->|Embedding Vectors| TransformerBlock[Transformer Layer Block x L <br> Hidden internal details]:::block
    
    %% Output
    TransformerBlock --> FinalNorm[Final RMSNorm]:::layer
    FinalNorm --> OutEmbed[Output Embedding / LM Head]:::layer
    OutEmbed -->|Logits| SoftmaxFinal[Softmax]:::math
    SoftmaxFinal --> NextToken([Next-Token Probability Distribution]):::data

    style TransformerBlock fill:#fff3e0,stroke:#f57c00,stroke-width:3px;
```

### Transformer Layer Block Internal Details

```mermaid
graph TD
    %% Color Palette Configurations
    classDef data fill:#e1f5fe,stroke:#039be5,stroke-width:2px;
    classDef layer fill:#ede7f6,stroke:#5e35b1,stroke-width:2px;
    classDef math fill:#f1f8e9,stroke:#558b2f,stroke-width:2px;

    %% Data Inputs
    Input([Input Hidden States]):::data --> RMS1[Pre-LN RMSNorm]:::layer
    Input -->|Residual Path| ResAdd1((+)):::math
    
    RMS1 --> AttnBlock[Causal Multi-Head Self-Attention with RoPE]
    AttnBlock --> ResAdd1
    
    ResAdd1 --> RMS2[Pre-LN RMSNorm]:::layer
    ResAdd1 -->|Residual Path| ResAdd2((+)):::math

    RMS2 --> FFNBlock[Position-Wise SwiGLU FFN]
    FFNBlock --> ResAdd2
    
    ResAdd2 --> Output([Output Hidden States]):::data

    style AttnBlock fill:#fce4ec,stroke:#c2185b,stroke-2px;
    style FFNBlock fill:#e8f5e9,stroke:#2e7d32,stroke-2px;
```

### Architectural Pipeline Flow

#### High-Level Pipeline Components

* **Raw Text Input**: The source sequence of natural language characters or words.

* [**BPE Tokenizer**](/LLM%20Architecture/1-tokenizer.md): Converts raw input text into a sequence of discrete integer tokens using Byte-Pair Encoding.

* [**Token Embedding Table**](/LLM%20Architecture/3-token-embedding.md): A lookup table that maps each token ID to a high-dimensional vector space representation.

* [**Transformer Layer Block**](/LLM%20Architecture/5-causal-multi-head-self-attention-with-rope.md) (× L): The core stack of $L$ repeated layers that iteratively transform the token representations.

* [**Final RMSNorm**](/LLM%20Architecture/4-rms-normalization.md): Applied after the final transformer block to stabilize gradients and normalize the final representations.

* [**Output Embedding / LM Head**](/LLM%20Architecture/8-output-embedding.md): Projects the normalized hidden states back into the vocabulary space to generate logits.

* **Softmax**: Converts the raw vocabulary logits into a normalized probability distribution.

* **Next-Token Probability Distribution**: The predicted probabilities for the subsequent token.

#### Transformer Layer Block Components

* **Input Hidden States**: The representation vector entering the block.

* [**Pre-LN RMSNorm (RMS1 & RMS2)**](/LLM%20Architecture/4-rms-normalization.md): Root Mean Square Normalization applied in a pre-layer normalization configuration (before the attention and feed-forward layers) to facilitate stable deep network training.

* [**Residual Paths & Addition Nodes ((+))**](/LLM%20Architecture/6-residule-path.md): Skip connections that add the inputs of a sub-layer back to its output, bypassing the transformations to preserve gradient flow.

* [**Causal Multi-Head Self-Attention with RoPE**](/LLM%20Architecture/5-causal-multi-head-self-attention-with-rope.md): Processes token-to-token dependencies. It uses a causal attention mask to preserve autoregressive properties and injects relative positional information using Rotary Position Embeddings (RoPE).

* [**Position-Wise SwiGLU FFN**](/LLM%20Architecture/7-position-wise-feed-forward.md): A feed-forward network block that uses the Gated Linear Unit with Swish activation (SwiGLU) for non-linear, high-capacity token representation transformations.

* **Output Hidden States**: The final transformed representation vector exiting the block.