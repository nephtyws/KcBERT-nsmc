import copy
import json
import logging
import os
import torch
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


class InputExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NSMCProcessor:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def _read_file(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f]

        return lines

    @staticmethod
    def _create_examples(lines, set_type):
        """ Creates an examples for the training and dev sets. """
        examples = []

        for (i, line) in enumerate(lines[1:]):
            line = line.split('\t')
            guid = f"{set_type}-{i}"
            text_a = line[1]
            label = int(line[2])

            if i % 1000 == 0:
                logger.info(line)

            examples.append(InputExample(guid=guid, text_a=text_a, label=label))

        return examples

    def get_examples(self, mode):
        file_to_read = None

        if mode == "train":
            file_to_read = self.args.train_file

        elif mode == "dev":
            file_to_read = self.args.dev_file

        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info(f"Looking at {os.path.join(self.args.data_dir, file_to_read)}")

        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode)


processors = {
    'nsmc': NSMCProcessor,
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")

        tokens = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:max_seq_len - special_tokens_count]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, f"Error with input length {len(input_ids)} vs {max_seq_len}"
        assert len(attention_mask) == max_seq_len, f"Error with attention mask length {len(attention_mask)} vs {max_seq_len}"
        assert len(token_type_ids) == max_seq_len, f"Error with token type length {len(token_type_ids)} vs {max_seq_len}"

        label_id = example.label

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"guid: {example.guid}")
            logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
            logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            logger.info(f"attention_mask: {' '.join([str(x) for x in attention_mask])}")
            logger.info(f"token_type_ids: {' '.join([str(x) for x in token_type_ids])}")
            logger.info(f"label: {example.label} (id = {label_id})")

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id
            )
        )

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_file_name = f"cached_{args.task}_{list(filter(None, args.model_name_or_path.split('/'))).pop()}_{args.max_seq_len}_{mode}"

    cached_features_file = os.path.join(args.data_dir, cached_file_name)

    if os.path.exists(cached_features_file):
        logger.info(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)

    else:
        logger.info(f"Creating features from dataset file at {args.data_dir}")

        if mode == "train":
            examples = processor.get_examples("train")

        elif mode == "dev":
            examples = processor.get_examples("dev")

        elif mode == "test":
            examples = processor.get_examples("test")

        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer)
        logger.info(f"Saving features into cached file {cached_features_file}")
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)

    return dataset
