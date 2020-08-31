## Discriminate Similar Languages  
### Overview
This repository reuses code from Huggingface transformers with minor changes to transformers/data/processors/glue.py and /transformers/data/metrics/\_\_init__.py for training BERT classifier. It also contains a self-implemented script (DSL_svm.py) for training SVM classifier. 

### Data
The experiments are trained and evaluated on  [DSLCC v4.0](http://ttg.uni-saarland.de/resources/DSLCC/)

### Train and evaluate with SVM
```
python3 DSL_svm.py
```
This will result in a 0.8845116836428999 accuracy.

### Train and evaluate with pre-trained multi-lingual BERT

For training, run
```
python3 run_dsl.py \ 
    --model_type bert \
    --model_name_or_path bert-base-multilingual-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/to/dslcc4/ \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --overwrite_output_dir \
    --task_name dsl \
    --save_steps 20000 \
    --output_dir ./dsl
```

For evaluating, run
``` 
python3 run_dsl.py \
    --model_type bert \
    --model_name_or_path bert-base-multilingual-uncased \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/to/dslcc4/ \
    --max_seq_length 512 \
    --task_name dsl \
    --output_dir ./dsl
```
This should result in a 0.9127142857142857 accuracy. 