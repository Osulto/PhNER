from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# Path to the text file
file_path = r"C:\Users\Jorolato\Desktop\SC 2024 Jan.txt"

# Step 1: Read the text from the file
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Step 2: Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)

# Step 3: Get the model outputs (logits)
outputs = model(**inputs).logits

# Step 4: Get the predicted token-level classes (highest score index)
predictions = torch.argmax(outputs, dim=2)

# Step 5: Map the predictions back to labels using the model's config
predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]

# Step 6: Print tokens with predicted labels (joining WordPiece tokens)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Initialize an empty string for joining subword tokens
final_tokens = []
current_word = ""

for token, label in zip(tokens, predicted_labels):
    # If the token starts with '##', it is a continuation of the previous word
    if token.startswith("##"):
        current_word += token[2:]  # Remove the "##" and append to current word
    else:
        # If there is a current word being built, add it to the final tokens
        if current_word:
            final_tokens.append((current_word, current_label))
        # Start a new word
        current_word = token
        current_label = label

# Add the last word to the final tokens
if current_word:
    final_tokens.append((current_word, current_label))

# Print the final tokens and their corresponding labels
for word, label in final_tokens:
    print(f"{word}: {label}")
