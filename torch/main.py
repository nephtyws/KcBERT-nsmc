import argparse

from torch.data_loader import load_and_cache_examples
from torch.trainer import Trainer
from torch.utils import init_logger, load_tokenizer, set_seed


def main(args):
    init_logger()
    set_seed(args)

    tokenizer = load_tokenizer(args)
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = None
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.load_model()
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="nsmc", type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--train_file", default="ratings_train.txt", type=str, help="Train file")
    parser.add_argument("--test_file", default="ratings_test.txt", type=str, help="Test file")

    parser.add_argument("--model_name_or_path", default="beomi/kcbert-base", type=str)

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=256, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=146, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=3000, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="Linear warmup over warmup_steps.")

    # logging_steps
    parser.add_argument('--eval_steps', type=int, default=500, help="Evaluate model every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", default=True, help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_false", default=False, help="Avoid using CUDA when available")

    args = parser.parse_args()

    main(args)
