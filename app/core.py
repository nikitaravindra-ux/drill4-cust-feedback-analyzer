"""Your code. One stub per milestone; build them in order and keep each one small."""

# --- Milestone 1: Define 3-5 feedback categories and create a small dataset of example comments with labels.
# Hint: Start by brainstorming common feedback types and then manually labeling a few examples.
def milestone_1() -> dict:
    """Build the labelled dataset (app/dataset.py): feedback.csv + stratified train/test split. Returns stats."""
    from app.dataset import build_dataset
    return build_dataset()


# --- Milestone 2: Extend GPT-2 tokenizer and model with new category tokens.
# Hint: Recall how special tokens were added to the tokenizer and model in the legal GPT drill.
def milestone_2(tokenizer, model) -> list[int]:
    """Add the category tokens to tokenizer + model, resize the embeddings (app/tokens.py). Returns their ids."""
    from app.tokens import add_category_tokens
    return add_category_tokens(tokenizer, model)


# --- Milestone 3: Fine-tune the model using the Hugging Face Trainer.
# Hint: Apply the `Trainer` class setup and training loop from the fine-tuning drill.
def milestone_3(**kwargs) -> dict:
    """Fine-tune with the Hugging Face Trainer (app/finetune.py) and save to models/. Returns training info."""
    from app.finetune import train
    return train(**kwargs)


# --- Milestone 4: Implement an inference function to classify new feedback.
# Hint: Adapt the prediction logic from the drill to generate category tokens.
def milestone_4(text: str, classifier=None):
    """Classify one comment (app/classify.py): one forward pass, argmax over the category-token logits."""
    from app.classify import load_classifier
    return (classifier or load_classifier()).classify(text)


# --- Milestone 5: Build a simple Streamlit app to input comments and show classifications.
# Hint: Use Streamlit widgets for text input and display, similar to previous app-building exercises.
def milestone_5(text: str, classifier) -> dict:
    """The Streamlit logic layer (app/ui_logic.py; the page itself is app/main.py): label + all scores."""
    from app.ui_logic import analyze_comment
    return analyze_comment(text, classifier)
