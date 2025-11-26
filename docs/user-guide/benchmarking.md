# Benchmarking

Evaluating your model is crucial to measure improvement.

## ROUGE

**Recall-Oriented Understudy for Gisting Evaluation**

*   **ROUGE-1**: Overlap of unigrams (single words).
*   **ROUGE-2**: Overlap of bigrams (two-word sequences).
*   **ROUGE-L**: Longest Common Subsequence.

ROUGE is excellent for summarization and translation tasks where there is a reference "gold standard" output.

## Other Metrics (Coming Soon)

*   **BLEU**: Standard for machine translation.
*   **BERTScore**: Semantic similarity using BERT embeddings.
*   **Perplexity**: Measure of how well the model predicts the sample text.

## The Comparison Step

The CLI automatically runs the selected benchmark on:
1.  The **Base Model** (before training).
2.  The **Fine-tuned Model** (after training).

It then displays a side-by-side comparison table showing the percentage improvement.
