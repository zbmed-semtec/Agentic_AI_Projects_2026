# Congi-BERT

- Model ID: adarshm09/congi-bert
- License: MIT
- Tasks: asia, categorical, nlp, deep learning, text, multiclass classification, transfer learning, python, bert, english
- Frameworks: PyTorch
- Shared by: Adarsh Maurya
- Source: https://www.kaggle.com/models/adarshm09/congi-bert

## Description

# Model Summary

Congi-BERT is fine-tuned BERT model based on transformer architecture which is fine-tuned for mutli-class classification on labelled dataset of dark pattern with 7 categorical mappings and Not Dark Pattern using pyTorch and wandb

## Usage

You can use this model by loading the model and then tokenizing the text input and it will predict the dark pattern category. Take reference from given code :
```python
# necessary libraries
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained model
label_dict = {"Urgency": 0, "Not Dark Pattern": 1, "Scarcity": 2, "Misdirection": 3, "Social Proof": 4, "Obstruction": 5, "Sneaking": 6, "Forced Action": 7}
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_dict))

# Load fine-tuned weights
fine_tuned_model_path = "models/finetuned_BERT_5k_epoch_5.model"
model.load_state_dict(torch.load(fine_tuned_model_path, map_location=torch.device('cpu')))

# Preprocess the new text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Function to map numeric label to dark pattern name
def get_dark_pattern_name(label):
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    return reverse_label_dict[label]

def find_dark_pattern(text_predict):
    encoded_text = tokenizer.encode_plus(
        text_predict,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    # Making the predictions
    model.eval()

    with torch.no_grad():
        inputs = {
            'input_ids': encoded_text['input_ids'],
            'attention_mask': encoded_text['attention_mask']
        }
        outputs = model(**inputs)

    predictions = outputs.logits

    # Post-process the predictions
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()

    return get_dark_pattern_name(predicted_label)
```

## Implementation requirements

GPU : P100 (16gb ram)
CPU: 30GB RAM
HDD: 20GB


# Model Characteristics

## Model initialization
Cogni-bert was fine tuned on BERT model by Google using pytorch and transformers

## Model stats
Size : 428mb


## Training data


## Demographic groups

Describe any demographic data or attributes that suggest demographic groups

## Evaluation data

250 separate samples with 7 dark pattern categories and Not dark Pattern 

Category
Urgency             56
Social Proof        52
Scarcity            49
Not Dark Pattern    34
Sneaking            31
Misdirection        30
Forced Action       26

# Evaluation Results

Results table

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16065374%2Ff0efca719097da6c671a72c052c91783%2FScreenshot%202024-05-16%20183159.png?generation=1715865050561599&alt=media)

Confusion Matrix:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16065374%2F5e193fb479b8804f8b3602100f9b35c1%2FScreenshot%202024-05-16%20183249.png?generation=1715865097008653&alt=media)



## Summary

Summarize and link to evaluation results for this analysis.
