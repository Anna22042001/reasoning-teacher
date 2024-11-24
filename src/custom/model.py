"""
LightningModule for training encoder OR encoder_decoder models which provides:
- Saving intermediate validation predictions as CompletionDataset
- Logging intermediate validation metrics (from the CompletionDataset)
"""
import copy
import json
import logging
from typing import List, Dict

import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import PreTrainedTokenizerBase
from custom.utils import invoke_tools

from data.completion_dataset import CompletionDataset, CompletionMetadata
from data.dataset import Dataset
from evaluation.evaluator import Evaluator
from evaluation.summary import summarize_evaluation


class Model(pl.LightningModule):
    validation_predictions: Dict

    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, model_type: str, use_cpu_offload=False,
                 completion_metadata: CompletionMetadata = None, lr=3e-4, truncate_early=True, max_length=1024):
        """
        - completion_metadata: metaddata used to save completions. If None, completions are not saved.
          `epoch_N` is appended to the `train_key` when saving intermediate validation completions.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.use_cpu_offload = use_cpu_offload
        self.completion_metadata = completion_metadata
        self.lr = lr
        self.max_length = max_length
        self.truncate_early = truncate_early

    def training_step(self, batch, batch_idx):
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        return self.model(**kwargs)["loss"]

    def validation_step_ori(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Returns outputs in dictionary format, since it's the only way that seems to work with `all_gather`
        """
        if self.current_epoch < 2 and self.truncate_early:
            max_length = 256
        else:
            max_length = self.max_length

        if self.model_type == "encoder_decoder":
            output = self.model.generate(batch["input_ids"], max_length=max_length).detach()
        elif self.model_type == "decoder":
            output = self.model.generate(batch["input_ids"], max_length=max_length,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id).detach()
        else:
            raise NotImplementedError("model_type='{}' not supported".format(self.model_type))

        return {
            "sample_index": batch["sample_index"],
            "input": batch["input_ids"],
            "output": output,
        }
    def validation_step_single_row(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Processes each row in the batch individually and returns outputs in dictionary format, 
        since it's the only way that seems to work with `all_gather`.
        """
        if self.current_epoch < 2 and self.truncate_early:
            max_length = 256
        else:
            max_length = self.max_length

        # Initialize lists to store each sample's results
        sample_indices = []
        inputs = []
        outputs = []

        # Iterate through each row in the batch
        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i].unsqueeze(0)  # Add batch dimension

            # Generate output based on model type
            if self.model_type == "encoder_decoder":
                padded_outputs = self.model.generate(input_ids, max_length=max_length).detach()
            elif self.model_type == "decoder":
                output = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                ).detach()
            else:
                raise NotImplementedError(f"model_type='{self.model_type}' not supported")

            # Append each result to the respective list
            sample_indices.append(batch["sample_index"][i])
            inputs.append(input_ids.squeeze(0))  # Remove batch dimension
            outputs.append(output.squeeze(0))    # Remove batch dimension

        max_output_length = max([output.size(0) for output in outputs])
        padded_outputs = torch.stack(
            [torch.nn.functional.pad(output, (0, max_output_length - output.size(0)),
                                    value=self.tokenizer.pad_token_id) for output in outputs]
        )

        # Return a dictionary of results with padded outputs
        return {
            "sample_index": batch["sample_index"],
            "input": batch["input_ids"],
            "output": padded_outputs,
        }

    def validation_step_single_token(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Processes each row in the batch individually and returns outputs in dictionary format, 
        since it's the only way that seems to work with `all_gather`.
        """
        if self.current_epoch < 2 and self.truncate_early:
            max_length = 128
        else:
            max_length = self.max_length

        # Initialize lists to store each sample's results
        sample_indices = []
        inputs = []
        outputs = []

        # Iterate through each row in the batch
        if self.model_type == "encoder_decoder":
            for i in range(len(batch["input_ids"])):
                input_ids = batch["input_ids"][i].unsqueeze(0)  # Add batch dimension
                attention_mask = batch["attention_mask"][i].unsqueeze(0)
                B, _ = input_ids.size()
                labels = torch.zeros(B, 1, dtype=torch.long, device=input_ids.device)
                encoder_outputs = None
                for _ in range(max_length):
                    out = self.model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=labels,
                    )
                    top_labels = out.logits[:, -1].argmax(-1).unsqueeze(-1)
                    labels = torch.cat([labels, top_labels], dim=-1)
                    if top_labels.item() == self.eos_token_id:
                        break
                    
                outputs.append(labels)
            
            max_output_length = max([output.size(0) for output in outputs])
            padded_outputs = torch.stack(
                [torch.nn.functional.pad(output, (0, max_output_length - output.size(0)),
                                        value=self.tokenizer.pad_token_id) for output in outputs]
            )
            # output = self.model.generate(batch["input_ids"], max_length=max_length).detach()
        elif self.model_type == "decoder":
            padded_outputs = self.model.generate(batch["input_ids"], max_length=max_length,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id).detach()
        else:
            raise NotImplementedError("model_type='{}' not supported".format(self.model_type))

        return {
            "sample_index": batch["sample_index"],
            "input": batch["input_ids"],
            "output": padded_outputs,
        }
    def validation_step_single_token_tool(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Processes each row in the batch individually and returns outputs in dictionary format, 
        since it's the only way that seems to work with `all_gather`.
        """
        if self.current_epoch < 2 and self.truncate_early:
            max_length = 128
        else:
            max_length = self.max_length

        # Initialize lists to store each sample's results
        sample_indices = []
        inputs = []
        outputs = []

        # Iterate through each row in the batch
        if self.model_type == "encoder_decoder":
            for i in range(len(batch["input_ids"])):
                input_ids = batch["input_ids"][i].unsqueeze(0)  # Add batch dimension
                attention_mask = batch["attention_mask"][i].unsqueeze(0)
                B, _ = input_ids.size()
                labels = torch.zeros(B, 1, dtype=torch.long, device=input_ids.device)
                encoder_outputs = None
                start = False
                end = False
                for _ in range(max_length):
                    out = self.model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=labels,
                    )
                    top_labels = out.logits[:, -1].argmax(-1).unsqueeze(-1)
                    labels = torch.cat([labels, top_labels], dim=-1)
                    if top_labels.item() == self.eos_token_id:
                        break
                    if top_labels.item() in [784, 636]:
                        start = True
                    elif top_labels.item() in [908, 4275, 908] and start:
                        end = True
                    if start and end:
                        to_tool = self.tokenizer.decode(labels[0])
                        tool_doned = invoke_tools(text = to_tool)
                        labels = self.tokenizer(tool_doned, return_tensors="pt")['input_ids'].to(input_ids.device)
                        start = False
                        end = False

                    
                outputs.append(labels)
            
            max_output_length = max([output.size(0) for output in outputs])
            padded_outputs = torch.stack(
                [torch.nn.functional.pad(output, (0, max_output_length - output.size(0)),
                                        value=self.tokenizer.pad_token_id) for output in outputs]
            )
            # output = self.model.generate(batch["input_ids"], max_length=max_length).detach()
        elif self.model_type == "decoder":
            padded_outputs = self.model.generate(batch["input_ids"], max_length=max_length,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id).detach()
        else:
            raise NotImplementedError("model_type='{}' not supported".format(self.model_type))

        return {
            "sample_index": batch["sample_index"],
            "input": batch["input_ids"],
            "output": padded_outputs,
        }

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Generates predictions one token at a time for each element in the batch.
        """

        return self.validation_step_ori(batch, batch_idx)
        
        # if self.current_epoch < 2 and self.truncate_early:
        #     max_length = 128
        # else:
        #     max_length = self.max_length

        # sample_indices = batch["sample_index"]
        # inputs = batch["input_ids"]
        # outputs = []

        # for i in range(inputs.size(0)):  # Process each input in the batch
        #     input_ids = inputs[i].unsqueeze(0)  # Select one input at a time
        #     generated_tokens = input_ids.clone()  # Initialize with the input sequence
        #     # Generate tokens one-by-one until max_length is reached
        #     # print("max_length - input_ids.size(1):", max_length - input_ids.size(1))
        #     for j in range(max_length - input_ids.size(1)):
        #         # print('i, j', i, j)
        #         output_token = self.model.generate(generated_tokens, max_length=generated_tokens.size(1) + 1,
        #                                            pad_token_id=self.tokenizer.pad_token_id,
        #                                            eos_token_id=self.tokenizer.eos_token_id).detach()
        #         # print('output_token', output_token.shape)
        #         output_token = output_token[:, -1:]
        #         # print('output_token', output_token.shape)
        #         # Append the new token to the sequence
        #         generated_tokens = torch.cat((generated_tokens, output_token), dim=1)

        #         # Stop if EOS token is generated
        #         if output_token.item() == self.tokenizer.eos_token_id:
        #             break
        #     outputs.append(generated_tokens)

        # # Stack outputs to ensure batch-wise consistency
        # max_generated_length = max([gen.size(1) for gen in outputs])

        # # Pad each sequence in outputs to the max_generated_length
        # padded_outputs = []
        # for gen in outputs:
        #     padding_length = max_generated_length - gen.size(1)
        #     if padding_length > 0:
        #         pad_token_id = self.tokenizer.pad_token_id
        #         padding = torch.full((1, padding_length), pad_token_id, dtype=torch.long, device=gen.device)
        #         gen = torch.cat((gen, padding), dim=1)
        #     padded_outputs.append(gen)

        # # Stack padded outputs to ensure batch-wise consistency
        # outputs = torch.cat(padded_outputs, dim=0)
        # # print('OUTPUT', type(outputs), outputs.shape)
        # # print(outputs)
        # return {
        #     "sample_index": sample_indices,
        #     "input": inputs,
        #     "output": outputs,
        # }


    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        """
        Gather outputs from all GPUs and save validation predictions as a CompletionDataset and
        log validation metrics.

        Note, `all_gather` *concatenates* tensors from all GPUs along the first dimension.
        """
        # Determine total sample count and local max input/output length
        local_max_output_length = 0
        local_max_input_length = 0
        total_samples = 0
        for batch in outputs:
            local_max_input_length = max(local_max_input_length, batch["input"].shape[-1])
            local_max_output_length = max(local_max_output_length, batch["output"].shape[-1])
            total_samples += batch["sample_index"].shape[0]

        # Determine global max input/output length
        max_input_length = self.all_gather(torch.tensor(local_max_input_length, dtype=torch.long)).max()
        max_output_length = self.all_gather(torch.tensor(local_max_output_length, dtype=torch.long)).max()

        # Create local padded tensors
        local_outputs: dict = {
            "sample_index": torch.ones((total_samples,), dtype=torch.long) * self.tokenizer.pad_token_id,
            "input": torch.ones((total_samples, max_input_length), dtype=torch.long) * self.tokenizer.pad_token_id,
            "output": torch.ones((total_samples, max_output_length), dtype=torch.long) * self.tokenizer.pad_token_id,
        }

        # Populate local tensors
        start_index = 0
        for i, batch in enumerate(outputs):
            batch_size = batch["sample_index"].shape[0]
            end_index = start_index + batch_size
            local_outputs["sample_index"][start_index:end_index] = batch["sample_index"]
            input_width = batch["input"].shape[-1]
            output_width = batch["output"].shape[-1]
            if self.model_type == "encoder_decoder":
                local_outputs["input"][start_index:end_index, :input_width] = batch["input"]
                local_outputs["output"][start_index:end_index, :output_width] = batch["output"]
            elif self.model_type == "decoder":
                output_only_width = output_width - input_width
                local_outputs["input"][start_index:end_index, :input_width] = batch["input"]
                local_outputs["output"][start_index:end_index, :output_only_width] = batch["output"][:, input_width:]
            else:
                raise NotImplementedError("model_type='{}' not supported".format(self.model_type))

            start_index = end_index

        global_outputs = self.all_gather(local_outputs)
        if self.global_rank == 0:
            if global_outputs["sample_index"].dim() == 2:  # world_size > 1
                global_outputs["sample_index"] = global_outputs["sample_index"].flatten(start_dim=0, end_dim=1)
                global_outputs["output"] = global_outputs["output"].flatten(start_dim=0, end_dim=1)
                global_outputs["input"] = global_outputs["input"].flatten(start_dim=0, end_dim=1)

            final_output = {
                "sample_index": global_outputs["sample_index"].tolist(),
                "input": self.tokenizer.batch_decode(global_outputs["input"], skip_special_tokens=True),
                "output": self.tokenizer.batch_decode(global_outputs["output"], skip_special_tokens=True),
            }

            if self.completion_metadata is not None:
                # Save outputs as CompletionDataset
                cd = self._generate_completion_dataset(self.completion_metadata, final_output, epoch=self.current_epoch)
                cd.save()

                # Log validation examples
                examples = []
                for i in cd.indices[:5]:
                    examples.append(cd[i])
                logging.info("VALIDATION_EXAMPLES".center(80, "-"))
                logging.info(json.dumps(examples, indent=4))

                # Log metrics
                evaluation = Evaluator.evaluate_completion_dataset(cd)
                summary = summarize_evaluation(evaluation)
                if summary:
                    for key, value in summary.items():
                        if key == "accuracy":
                            self.log(key, value, prog_bar=True, logger=True)
                        else:
                            self.log(key, value, prog_bar=False, logger=True)

    def configure_optimizers(self):
        if self.use_cpu_offload:
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def _generate_completion_dataset(completion_metadata, output: Dict[str, List], epoch=None,
                                     completions_per_sample=1) -> CompletionDataset:
        """
        Initialize and populate a CompletionDataset from model output.

        - output: {
            sample_index: List[int],
            input: List[str],
            output: List[str],
          }
        - completions_per_sample: limit the number of completions used per sample. This is useful when model output
          is obtained from distributed inference, where some samples may be duplicated to match batch sizes. Will use
          all completions if None. Existing completions count towards the limit.
        """
        if completions_per_sample is not None and completions_per_sample < 1:
            raise ValueError("completions_per_sample must be at least 1")

        # Add/assign epoch to train key of completion_identifier
        completion_metadata = copy.deepcopy(completion_metadata)
        if epoch is not None:
            completion_metadata.epoch = epoch

        # Initialize completion dataset
        cd = CompletionDataset.init(completion_metadata)

        # Populate completion dataset with model output
        dataset = Dataset.load(cd.dataset_key)
        for sample_index, input, output in zip(output["sample_index"], output["input"], output["output"]):
            if len(dataset) <= sample_index:
                raise KeyError(
                    "Sample index {} not found in dataset {}".format(sample_index, cd.dataset_key))

            if sample_index in cd.data:
                completions = cd.data[sample_index]
            else:
                completions = list()
                cd.data[sample_index] = completions

            completion_index = len(completions)
            if completions_per_sample is None or completion_index < completions_per_sample:
                completions.append({
                    "sample_index": sample_index,
                    "completion_index": completion_index,
                    "question": dataset[sample_index]["question"],
                    "answer": dataset[sample_index]["answer"],
                    "prompt": input,
                    "completion": output,
                })
            cd.data[sample_index] = completions

        return cd
