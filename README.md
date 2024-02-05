# Evaluation quantized LLM locally. Bias. Hallucination. Toxicity.

## Create env and install requirements

```bash
python3.10 -m venv venv
source venv/bin/activate
make install
```

Download quantized Mistral-7B in GGUF format from [TheBloke/Mistral-7B-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/tree/main)


Set up path to save JSON results.

```bash
export DEEPEVAL_RESULTS_FOLDER="<YOUR_PATH>"
```

Run evaluation.
```bash
deepeval test run test_evaluation.py
```

In the terminal you should see mth like that.

![img.png](images/img.png)
