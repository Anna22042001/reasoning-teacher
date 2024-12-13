# Large Language Models Are Reasoning Teachers

## Instructions to Replicate Our Results (Team 04)

To replicate the results of the original paper, please follow the instructions below (starting after the "Getting Started" section).

### Our New Implementation Includes:

1. **Fine-tuning Data Preparation**
   - **Script:** `./reasoning-teacher/calculator/create_fine_tuning_with_calculator.py`
   - **Purpose:** Replaces all mathematical operations with calculator calls to construct new fine-tuning data for small models.
   - **Usage:** Requires the `file_path` to the file containing the original reasoning steps produced by the teacher models (e.g., `./D_multiarith.json` for the MultiArith dataset).

2. **Inference Function Processing**
   - **File:** `./reasoning-teacher/src/custom/utils.py`
   - **Purpose:** Processes function calls during inference (e.g., converting `Calculator(1+5*3+16/4)` into `20`).

3. **New Inference Code for Validation**
   - **Function:** `def validation_step_single_token_tool()` (located in `./reasoning-teacher/src/custom/model.py`)
   - **Purpose:** Enables function calling during validation. This function is only used within `def validation_step()` from the same file when running `./reasoning-teacher/scripts/custom/example_ft5_test.sh`.
   - **Note:** During training, the original inference function (`def validation_step_ori()`) is still used.

4. **Note**
   - **Running `example_ft5_test.sh`:** When running the test files after fine-tuning, the path to the checkpoint must be modified to match the dataset used. For example:
     ```python
     checkpoint = torch.load('/content/reasoning-teacher/external_lightning_logs/flan_t5_base_addsub_ft_cot/lightning_logs/version_0/checkpoints/epoch=14-step=405.ckpt')
     ```
     This example uses a checkpoint for `flan-t5-base` fine-tuned on the AddSub dataset.
   - Unlike the training script (`./reasoning-teacher/scripts/custom/example_ft5.sh`), which can handle multiple datasets, the test script (`./reasoning-teacher/scripts/custom/example_ft5_test.sh`) can only be run for one dataset (corresponding to a specific checkpoint) at a time.
   - The notebook to ran the whole training and inference pipeline on Google Colab is at `./reasoning-teacher/reasoning.ipynb`
   - We added three fine-tuning datasets (in both the original and augmented with calculator calls versions) used in our project (AddSub, MultiArith, SingleEq) under `./reasoning-teacher/calculator/`, so those datasets could be directly used for fine-tuning
   - All implementation is available at `https://github.com/Anna22042001/reasoning-teacher/tree/submit`
  
The original work is [Large Language Models Are Reasoning Teachers](https://arxiv.org/abs/2212.10071), by
Namgyu Ho, Laura Schmid, and Se-young Yun.


## Getting Started
This repository contains code for (1) running CoT reasoning on OpenAI models,
and (2) apply Fine-tune-CoT to train students based on OpenAI models *or* custom open-source models such as T5, Flan-T5, GPT-2 on your GPUs, based on 🤗 and Pytorch Lightning.

### OpenAI API Experiments

OpenAI API experiments are implemented in the `oai` module. Refer to `notebooks/example_oai_finetune_cot.ipynb`
on how to run Fine-tune-CoT from start to finish.

### Custom Experiments (on GPU) 

Custom experiments are implemented in the `custom` module, based on PyTorch Lightning. Refer to `custom_train.py`
and `scripts/custom/*.sh` on how to fine-tune models such as T5, Flan-T5, and GPT-2 using Fine-tune-CoT.

## Setup

```
pip install -r requirements.txt
python setup.py develop
```

### Environment

The code has been tested on Python<=3.10, PyTorch Lightning<=1.9, PyTorch>=2.0

## Data 🚀

We're proud to share *all* of our raw experimental data! All data is organized in json or jsonl format, for your pleasure :)

Cloud storage folder links:

- [Dropbox](https://www.dropbox.com/sh/hwcncpyomx87h20/AACqgVdd-ZzBQ3ncJcKqw0cVa?dl=0)
- [Google Drive](https://drive.google.com/drive/folders/1C6kah3WV36N8omlUl-TeU9tsJADZNaJV?usp=share_link)

### File List

- `dataset.tar.gz`: 12 task datasets compiled in a unified json format
  - Belongs in `PROJECT/data/dataset/`
- `completion_data.tar.gz`: Completion data, i.e., inference data, from all teachers and students, for *all* experiments. About 8GB when uncompressed
  - Belongs in `PROJECT/saved/completion_data/`
- `teacher_completion_data.tar.gz`: Completion data from Zero-shot-CoT (with diverse reasoning) on the default teacher model `text-davinci-002` using the OpenAI API. About 💰 $1000+ worth of goods, with ❤️ from [OSI LAB](http://osi.kaist.ac.kr) at [KAIST](https://kaist.ac.kr) . Subset of `completion_data.tar.gz`.
  - Belongs in `PROJECT/saved/completion_data/`.
- `finetune_data.tar.gz`: *All* data used to fine-tune OpenAI students via the fine-tuning API, in jsonl format. These are derived from teacher completion data and can be generated from our code.
  - Belongs in `PROJECT/saved/finetune_data/`

### Generate Paper Results

After downloading the full `completion_data.tar.gz`, you can run `notebooks/results.ipynb` to generate *all* result tables and figures from our paper. The code will (re-)evaluate all raw text model outputs contained in the completion data.



## Additional Resources

### Template-based Split (Paper Appendix E.3)

Template-based splits for MultiArith and Date Understanding are saved in `/data/splits/*__template.json`

### Few-shot Prompts

Few-shot prompts adapted from Wei 2022 are saved in `/data/few_shot_cot_prompts.json`



## Data Structures

### `data.dataset.Dataset`

```json
{
  "metadata": {
    "dataset_key": "multiarith"
  },
  "data": [
    {
      "sample_index": 0,
      "question": "string",
      "answer": "string",
      "rationale": "string?"
    }
  ]
}
```

### `data.completion.CompletionDataset`

```json
{
  "metadata": {
    "dataset_key": "multiarith",
    "base_model": "curie",
    "finetune_key": "zs_cot_multiarith",
    "train_key": "ft_cot",
    "prediction_template": "ft_cot_token",
  },
  "data": {
    "<sample_index>": [
      {
        "sample_index": 0,
        "completion_index": 0,
        "question": "string",
        "answer": "string",
        "prompt": "string",
        "completion": "string",
        "finish_reason": "string",
        "reasoning_prompt": "string?",
        "reasoning_completion": "string?",
        "reasoning_finish_reason": "string?",
      }
    ]
  }
}
```



## Data Organization

*Needs update.*

- `<model_key>` = `B_<base_model>_T_<train_key>`

### File Organization Pattern

```
saved/
|–– completion_data/
    |–– B_<BASE_MODEL>__C_<COMPLETION_KEY>/
        |-- D_<DATESET_KEY>.json  # base model inference
        |-- F_<FINETUNE_KEY>__D_<DATESET_KEY>.json  # default fine-tuned model inference
        |-- F_<FINETUNE_KEY>__T_<TRAIN_KEY>__D_<DATESET_KEY>.json  # custom fine-tuned model inference
|–– finetune_data/
    |–– P_<PLATFORM_KEY>/
        |–– F_<FINETUNE_KEY>{.*|/}
|–– model_metadata/
    |–– B_<base_model>
        |–– F_<FINETUNE_KEY>__T_<train_key>.json
```

### File Organization Examples

```
saved/
|–– completion_data/
    |–– B_text-davinci-002__C_zs_cot/
    |–– B_text-davinci-002__C_zs_cot_long/
    |–– B_text-davinci-002__C_fs_cot/
    |–– B_curie__C_zs_cot/
    |–– B_curie__C_fs_cot/
    |–– B_curie__C_zs/
    |–– B_curie__C_ft_cot/
|–– finetune_data/
    |–– F_zs_cot_multiarith/  # text-davinci-002_zs_cot
    |–– F_zs_cot_long_multiarith/
|–– model_metadata/
    |–– B_curie/
        |–– F_zs_cot_multiarith.json
```


### Personal Note

![accepted](acl2023.jpg)

