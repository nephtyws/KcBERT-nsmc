from flask import current_app, Flask, request
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from bert.utils import load_tokenizer


app = Flask(__name__)

args = torch.load(os.path.join("../bert/model", 'training_args.bin'))
model = AutoModelForSequenceClassification.from_pretrained("../bert/model")
model.to("cpu")
model.eval()
tokenizer = load_tokenizer(args)


@app.route("/")
def hello():
    return current_app.send_static_file("hello.html")


def predict(sentence):
    # sentence = request.form.get("sentence")

    def convert_input_to_tensor_dataset():
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token_id = tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []

        tokens = tokenizer.tokenize(sentence)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [0] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [0] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([1] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

        return dataset

    dataset = convert_input_to_tensor_dataset()
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=128)

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to("cpu") for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'labels': None,
            }

            outputs = model(**inputs)
            logits = outputs[0]

            confidence = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            label = np.argmax(confidence, axis=1)

    confidence = f"{max(confidence[0]):.3f}"
    sentiment = "positive" if label[0] else "negative"

    return f"Input: {sentence} and it's sentiment is {sentiment} with confidence {confidence}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
