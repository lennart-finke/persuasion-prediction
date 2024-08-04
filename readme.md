# Predicting Persuasing Scores

This repository provides an experimental setup to predict persuasiveness with language models, on a dataset from the study [Measuring the Persuasiveness of Language Models](https://www.anthropic.com/news/measuring-model-persuasiveness).

### How to use
To run an evaluation of persuasiveness prediction for a given model, we use the [Inspect](https://inspect.ai-safety-institute.org.uk) framework. After installing that, you can run `inspect eval persuasion.py@persuasion` to test the solver specified in `persuasion.py`. This creates logs you can analyze with `find_correlations.py`. A persuasiveness prediction finetuning dataset in the OpenAI API format can be generated with `finetuning.py`.

### Attributions
Many thanks to Durmus et al. for publishing their data.
