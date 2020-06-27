## 任务
GLUE Benchmark中的CoLA Task。  
The task is to determine the acceptability (grammaticality) by their original authors. (0 (unacceptable) / 1(unacceptable))
详见[PPT](https://github.com/Huntersxsx/SJTU-NLU2020-CoLA/blob/master/Assignment-CoLA.pptx)。


## 方法  

### 数据集
利用download_glue_data.py文件下载GLUE数据集。(下载过程需要VPN，这里我已经将下载好的CoLA数据集放入 ./glue_data/中) 

```
python download_glue_data.py --data_dir glue_data --tasks all
```

### BERT、RoBERTa、ALBERT

参考了[huggingface](https://github.com/huggingface/transformers)的代码，[安装教程](https://huggingface.co/transformers/installation.html)。  
主要使用了./Huggingface/examples/text-classification/中的run_prediction.py代码。  

原代码中的run_glue.py仅支持输出验证集的得分，这里依照[JetRunner](https://github.com/JetRunner/BERT-of-Theseus/tree/master/glue_script)的操作，来实现test.tsv的结果预测。 
首先，根据教程安装好transformers包，然后将原包中的transformers/data/processor/glue.py替换为本文件中的./Huggingface/glue.py文件，并用run_prediction.py代码替代原来的run_glue.py代码。
运行run_prediction.py代码。(这里要注意的是如果在本机跑，程序会自动去下载预训练好的模型，如果不开VPN可能会下载不成功，下载后的预训练模型会自动保存在C:\Users\username\.cache\torch\transformers目录中，也可加--cache_dir将其保存到其他目录中)。    

```
python F:/SJTU-NLU2020-CoLA/Huggingface/examples/text-classification/run_prediction.py \
    --model_type bert --model_name_or_path bert-base-cased --task_name CoLA \ 
    --do_train --do_eval --evaluate_during_training \
    --data_dir F:/SJTU-NLU2020-CoLA/glue_data/CoLA/ \
    --max_seq_length 80 --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=32 \
    --learning_rate 2e-5 --num_train_epochs 3.0 \
    --output_dir F:/SJTU-NLU2020-CoLA/cola_output/bert-base-cased/ \
    --logging_steps 400 --save_steps 400
```

在微调Large模型时，若把Batch Size提高到32，可能会出现OOM的错误，可以使用多卡微调，量化或者apex加速等操作。  

部分实验结果如下表：  

| Model | Matthew's Corr  |
| :---: | :---: |
| BERT-Base-uncased | 43.4 |
| BERT-Base-cased | 51.8 |
| RoBERTa-Base | 59.3 |
| BERT-Large-cased | 62.3 |
| RoBETRa-Large | 61.4 |
| ALBERT-Large vv2 | 61.0 |

由于时间限制，没有对RoBERTa和ALBERT进行进一步微调，所以获得结果不是很理想，此外，我还微调了XLNet，但是效果很差，于是我将更多精力放在ELECTRA的精调上。

### ELECTRA

参考了[lonePatient](https://github.com/lonePatient/electra_pytorch)的代码，[paper](https://openreview.net/pdf?id=r1xMH1BtvB)。  
原代码中的run_classifier.py仅支持输出验证集的得分，这里借鉴了[JetRunner](https://github.com/JetRunner/BERT-of-Theseus/tree/master/glue_script)处理使用Huggingface Transformers预测GLUE验证集的操作，增加了预测CoLA任务中test.tsv结果的代码，见predict_cola.py，并且修改了./processor/task_processor.py中ColaProcessor类，使代码可以预测test.tsv的结果。  

首先，下载[预训练好的ELECTRA模型](https://github.com/google-research/electra)，放到./prev_trained_model/对应的文件夹中，跟随lonePatient的[Fine-tune](https://github.com/lonePatient/electra_pytorch#fine-tuning)步骤，执行convert_electra_tf_checkpoint_to_pytorch.py代码

```
python convert_electra_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=./prev_trained_model/electra_base \
    --electra_config_file=./prev_trained_model/electra_base/config.json \
    --pytorch_dump_path=./prev_trained_model/electra_base/pytorch_model.bin
```  

然后执行predict_cola.py代码进行test.tsv的结果预测

```
python predict_cola.py --model_type=bert \
    --model_name_or_path=F:/SJTU-NLU2020-CoLA/ELECTRA/prev_trained_model/electra_base \
    --task_name="cola" --do_train --do_eval --do_predict --do_lower_case \
    --data_dir=F:/SJTU-NLU2020-CoLA/glue_data/CoLA/ \
    --max_seq_length=80 --per_gpu_train_batch_size=32 --per_gpu_eval_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --logging_steps=268 --save_steps=268 \
    --output_dir=F:/SJTU-NLU2020-CoLA/cola_output/electra-base80-2e5-seed50/ \
    --overwrite_output_dir --seed=50
```  

微调过程很快，ELECTRA-Base在仅使用CPU的情况下3个epoch在20多分钟就可以完成，ELECTRA-Large需要使用GPU，在1080ti环境下5分钟左右就可以微调完3个epoch。  

部分实验结果如下表：  

| Model | Matthew's Corr  | Seed |
| :---: | :---: | :---: |
| ELECTRA-Base | 66.2 | 42 |
| ELECTRA-Base | 65.4 | 36 |
| ELECTRA-Base | 66.4 | 50 |
| ELECTRA-Base | 66.5 | 64 |
| ELECTRA-Large | 67.8 | 42 |
| [ELECTRA-Large](https://pan.baidu.com/s/1siiKqY_WxIalJkMBRSzpYQ) | 71.9 | 50 |
| ELECTRA-Large | 70.4 | 64 |

提取码：rvyr。  
以上结果都是单模型，没有使用ensemble等技巧。

## 结果  
这里仅展示了最佳结果，所有结果的截图可见[Report.pdf](https://github.com/Huntersxsx/SJTU-NLU2020-CoLA/blob/master/Report.pdf)文件中。  
表现最好的单模型是用ELECTRA-Large微调得到的，微调参数为：Max Seq Length=80, Learning Rate=2e-5, Batch Size=32, Epochs=3.0, Seed=50。结果如下：  

![](https://github.com/Huntersxsx/SJTU-NLU2020-CoLA/blob/master/img/ELECTRA-large.png)


## 参考

Huggingface Transformers: https://github.com/huggingface/transformers  
electra_pytorch: https://github.com/lonePatient/electra_pytorch  
predict test.tsv: https://github.com/JetRunner/BERT-of-Theseus/tree/master/glue_script  
electra pre-trained model: https://github.com/google-research/electra  
download GLUE data: https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
