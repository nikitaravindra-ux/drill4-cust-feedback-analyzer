# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/nv9mdfwfbw/badge.svg)](https://ask.thili.ai/b/nv9mdfwfbw)

This project develops a Customer Feedback Analyzer for retail managers. It leverages a fine-tuned GPT-2 model to automatically categorize customer comments, proving the fine-tuning logic and inference pipeline from the legal GPT drill to make predictions.

## What you are building

**Goal.** You will build a tool that automatically categorizes customer feedback comments into predefined categories like "product issue" or "staff praise" using a fine-tuned GPT-2 model.

**Done when.** You will have a fine-tuned GPT-2 model capable of classifying new customer comments by generating special label tokens. The project is complete when you can input a customer comment into a Streamlit application and see its classification displayed.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**  
      Hint: Start by brainstorming common feedback types and then manually labeling a few examples.
- [ ] **2. Extend GPT-2 tokenizer and model with new category tokens.**  
      Hint: Recall how special tokens were added to the tokenizer and model in the legal GPT drill.
- [ ] **3. Fine-tune the model using the Hugging Face Trainer.**  
      Hint: Apply the `Trainer` class setup and training loop from the fine-tuning drill.
- [ ] **4. Implement an inference function to classify new feedback.**  
      Hint: Adapt the prediction logic from the drill to generate category tokens.
- [ ] **5. Build a simple Streamlit app to input comments and show classifications.**  
      Hint: Use Streamlit widgets for text input and display, similar to previous app-building exercises.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to upload a file (e.g., CSV) with multiple feedback comments for batch classification.
- Display a simple dashboard in Streamlit showing the count of comments per category for batch inputs.
- Add functionality to save classified feedback comments to a local file.
