# entexBERT

Repository containing code for training entexBERT, a fine-tuned DNABERT model with a classifier for the prediction of allele-specific behavior.

The model adds a fully-connected neural net layer on top of the token/sequence embedding of DNABERT to predict whether the single-nucleotide polymorphism located at the center of the window is sensitive to allele-specific effects. The pre-trained weights from DNABERT are used for entexBERT.

## Requirements

This project builds off of [DNABERT](https://github.com/jerryji1993/DNABERT), which should be installed according to the instructions found at the linked repository. Additionally, we use `pytorch==1.10.2` and `cudatoolkit==11.3.1`.

## Usage

Save the `.py` files from this repository in the same directory as the DNABERT training and finetuning scripts. Sample bash scripts for [training/fine-tuning](#fine-tuning), [testing](#testing), and [visualizing](#visualization) model outputs are provided below.

### Data

Each of `train.txt`, `test.txt`, and `dev.txt` should contain rows where each row consists of a sequence in kmer format followed by a (0/1) label.

Refer to the sample data directory `/model/` in this repository for the format required by the model. The DNABERT repository also provides [some sample data](https://github.com/jerryji1993/DNABERT/blob/master/examples/sample_data/pre/6_3k.txt)

The sample provided in this repository as well as data used for the paper is generated from the [EN-TEx data set](https://www.encodeproject.org/search/?type=Reference&internal_tags=ENTEx).

### Model

Pre-trained DNABERT models can be found [here](https://github.com/jerryji1993/DNABERT?tab=readme-ov-file#32-download-pre-trained-dnabert). Then, fine-tune according to instructions in [this section](#fine-tuning).

### Fine-tuning

`KMER`: 3, 4, 5 or 6

`MODEL_PATH`: where the pre-trained DNABERT model is located

`DATA_PATH`: where the train, test, and dev data sets are stored

Set other hyperparameters according to your use case.

```bash
python3 entexbert_ft.py \
    --model_type ${model} \
    --tokenizer_name=dna$KMER \
    --model_name_or_path \$MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir \$DATA_PATH \
    --predict_dir \$DATA_PATH \
    --max_seq_length ${seq_len} \
    --per_gpu_eval_batch_size=${batch}   \
    --per_gpu_train_batch_size=${batch}   \
    --learning_rate ${lr} \
    --num_train_epochs ${ep} \
    --output_dir \$OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 5000 \
    --save_steps 20000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8 \
    --pred_layer ${layer} \
    --seed ${seed}
```

### Testing

`PREDICTION_PATH` specifies where you would like to store predictions.

```bash
python3 entexbert_ft.py \
    --model_type ${model} \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length ${seq_len} \
    --per_gpu_pred_batch_size=${batch}   \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 8
```

### Visualization

```bash
python3 entexbert_ft.py \
    --model_type ${model} \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_visualize \
    --visualize_data_dir $DATA_PATH \
    --visualize_models $KMER \
    --data_dir $DATA_PATH  \
    --max_seq_length ${seq_len} \
    --per_gpu_pred_batch_size=${batch}   \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 8
```
