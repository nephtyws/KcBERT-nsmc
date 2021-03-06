import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, BertForSequenceClassification

from bert.utils import compute_metrics, get_label


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.labels = get_label()
        self.num_labels = len(self.labels)

        self.config_class = AutoConfig
        self.model_class = BertForSequenceClassification

        self.config = self.config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(self.labels)},
            label2id={label: i for i, label in enumerate(self.labels)},
        )

        self.model = self.model_class.from_pretrained(args.model_name_or_path, config=self.config)

        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                        len(train_dataloader) // self.args.gradient_accumulation_steps) + 1

        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        logger.info("***** Running training *****")
        logger.info(f"Num examples = {len(self.train_dataset)}")
        logger.info(f"Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"Total train batch size = {self.args.train_batch_size}")
        logger.info(f"Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"Total optimization steps = {t_total}")
        logger.info(f"Eval steps = {self.args.eval_steps}")
        logger.info(f"Save steps = {self.args.save_steps}")

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3],
                }
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.eval_steps > 0 and global_step % self.args.eval_steps == 0:
                        self.evaluate("test", global_step)  # Only test set available for NSMC

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode, global_step=0):
        if mode == 'test':
            dataset = self.test_dataset

        elif mode == 'dev':
            dataset = self.dev_dataset

        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        logger.info(f"***** Running evaluation on {mode} dataset *****")
        logger.info(f"Num examples = {len(dataset)}")
        logger.info(f"Batch size = {self.args.eval_batch_size}")
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3],
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()

            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        results = {
            'loss': eval_loss
        }

        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results *****")
        with open("./model/evaluation_result.txt", "a+", encoding="utf-8") as f:
            for key in sorted(results.keys()):
                logger.info(f"{key} = {str(results[key])}")
                f.write(f"Current global step: {global_step} and results: {key} = {str(results[key])}\n")

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info(f"Saving model checkpoint to {self.args.model_dir}")

    def load_model(self):
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")

        except:
            raise Exception("Some model files might be missing...")
