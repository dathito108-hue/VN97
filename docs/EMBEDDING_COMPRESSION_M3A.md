# VN97 M3A factorized tied embeddings

VN97 keeps tied input embedding and output language-head weights as a canonical invariant.
M3A adds an optional low-rank representation without changing that invariant.

## Compatibility mode

When VN97Config.embedding_rank is None, VN97LanguageCore uses the original representation:

    W: [V, D]
    embedding(token) = W[token]
    logits(h) = h @ W^T

The nn.Embedding weight object is still assigned directly to the bias-free nn.Linear language
head. This is the default, so existing M0-M2 reference behavior and checkpoints do not silently
change.

## Factorized mode

For embedding rank R with 0 < R < D:

    E: [V, R]
    P: [R, D]
    W_effective = E @ P

Input lookup is:

    embedding(token) = E[token] @ P

Output logits are computed without materializing W_effective:

    reduced = h @ P^T
    logits = reduced @ E^T

FactorizedEmbedding and FactorizedLMHead reference the exact same E and P Parameter objects.
There is no independent output-head copy.

## Parameter count

Full tied representation:

    V * D

Factorized tied representation:

    V * R + R * D

For V=32,000, D=512 and R=64:

    full       = 16,384,000 parameters
    factorized =  2,080,768 parameters

The factorized representation uses about 12.7% of the vocabulary parameters, a reduction of
about 87.3%. This is a parameter/storage comparison, not a device latency claim; device-specific
latency belongs to later runtime benchmarking.

## Numerical contract

effective_weight() exists for reference tests. M3A tests require:

    FactorizedEmbedding(ids)
        == embedding(ids, E @ P)

and:

    FactorizedLMHead(h)
        == linear(h, E @ P)

within floating-point tolerance.

The recurrent-language-core invariant also remains unchanged: full-sequence execution must
match repeated token-step execution in both compatibility and factorized modes.

## Checkpoint boundary

Full and factorized modes have different parameter structures. Conversion between a trained
full matrix and a selected low rank is an explicit export/adaptation operation; setting
embedding_rank never silently approximates an existing full checkpoint.
