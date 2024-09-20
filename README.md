# Viper: Open Mamba-based Vision-Language Models
**Yufan Zhuang<sup>1,2</sup>, Pierce Chuang<sup>2</sup>, Yichao Lu<sup>2</sup>, Abhay Harpale<sup>2</sup>, Vikas Bhardwaj<sup>2</sup>, Jingbo Shang<sup>1</sup>**

**<sup>1</sup>UC San Diego**, **<sup>2</sup>Meta**

[Viper-Jamba-52B](https://huggingface.co/ViperVLM/Viper-Jamba-52B) || [Viper-Mamba-7B](https://huggingface.co/ViperVLM/Viper-Mamba-7B) || [Evaluation](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) || [Github](https://github.com/EvanZhuang/viper)

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/6438ccbb3b46237de3d052e8/RFArMOH2TMI_G9bZTZr8_.jpeg)
(Logo Created by ChatGPT-4o)


* Viper VLMs are built on the Mamba architecture, which offers efficiency and strong performance in handling long-range dependencies compared to Transformers.
* The models process visual tokens from entire images, leveraging Mamba's strengths in linear-time complexity and long-range reasoning for vision tasks, and are trained on the Cambrian-7M dataset, supporting up to 2K resolution.
* Viper VLMs demonstrate competitive performance on diverse benchmarks, setting the stage for potential future shifts in vision-language model architectures.

## Introduction

We introduce *Viper*, a series of open vision language models (VLMs) built on the Mamba architecture.
Since Mamba's inception, it has been regarded as a promising alternative to the Transformer as the foundational architecture for large language models.
Mamba offers a significant advantage in terms of linear-time complexities with respect to input sequence length, while also outperforming Transformers in tasks that require long-range dependencies understanding.

In Viper VLMs, we imbibe all visual tokens into the model and inference on the entire image, relying on Mamba's efficiency and long-range reasoning power to comprehend the vision inputs.
The models are trained on the Cambrian-7M, natively supporting up to 2K resolution.
We show that Viper VLMs are competitive with open-sourced VLMs across diverse benchmarks.
This work lays the groundwork for potential architectural shifts in future vision-language models, highlighting Mamba's promising role in advancing this field.



## Model Architecture

We use the single-encoder design with linear projectors connecting the vision encoder and LLM backbones.

| Model | Encoder | LLM backbone| Arch | Input Resolution (Training)
|----------|----------|----------|----------|----------|
| Viper-Jamba-52B | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | [Jamba-1.5-Mini](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini) | MoE-Jamba | Up to 1344x1344 pixels |
| Viper-Mamba-7B  | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | [falcon-mamba-7b-instruct](tiiuae/falcon-mamba-7b-instruct) | Dense-Mamba | Up to 2352x2352 pixels|

We utilized AnyRes for supporting high-resolution inputs.


## Evaluation


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6438ccbb3b46237de3d052e8/qs5uJXAgUUE1qL1XeWghH.png)


## Usage

Environment Configuration
```
git clone https://github.com/EvanZhuang/viper.git
cd ./viper
```
Create conda environment
```
conda create --name viper python=3.10
conda activate viper
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install mamba-ssm[causal-conv1d]
```
Dependent on [flash-attn](https://github.com/Dao-AILab/flash-attention), [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d), [mamba-ssm](https://github.com/state-spaces/mamba)

First install from here:
```
pip install vipervlm
```
Then you can use the Viper VLMs in the following way:
```
import copy
import torch
from viper.model.builder import load_pretrained_model
from viper.conversation import conv_templates
from viper.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token

model_path = ""
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, use_flash_attn=True)
model.eval()

conv_mode = 'system_jamba'
DEFAULT_IMAGE_TOKEN = '<image>'
IMAGE_TOKEN_INDEX = -200

content, images = '', []
image_sizes = []  # Store image sizes

# Process Input in Chat format
for msg in message:
    if msg['type'] == 'text':
        content += msg['value']
    else:
        img = Image.open(msg['value']).convert('RGB')
        images.append(img)
        image_sizes.append(img.size)  # Store the size of each image
        content += (DEFAULT_IMAGE_TOKEN + '\n')

# Process images using the class attribute process_images
image_tensor = process_images(images, image_processor, model.config)[0]

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], content)

prompt_question = conv.get_prompt(add_generation_prompt=True)

input_ids = tokenizer_image_token(prompt_question,
                                       tokenizer,
                                       IMAGE_TOKEN_INDEX,
                                       return_tensors='pt')
input_ids = input_ids.unsqueeze(0).to(device='cuda', non_blocking=True)
image_tensor = image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)

# Pass image sizes along with other parameters
with torch.inference_mode():
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        max_new_tokens=4096,
        temperature=0,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

```

## Throughput Analysis
Viper-Jamba-52B's active parameter size is only 12B.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6438ccbb3b46237de3d052e8/16SScCnjiMMMBkibRd7JD.png)

## Dataset
We train our models on [Cambrian-7M](https://github.com/cambrian-mllm/cambrian).
These datasets provide a wide variety of high-quality image-conversation pairs sourced from diverse environments and contexts, enabling robust multi-modal learning. 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6438ccbb3b46237de3d052e8/xgK6Bg8TuFbWzB4BephZn.png)

## Training Recipe
We employ a progressive three-stage training procedure designed to optimize performance across varying levels of input complexity and resolution. 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6438ccbb3b46237de3d052e8/vQHSIf3PRYab1g8c-owzJ.png)

The training process begins with low-resolution inputs, allowing the model to focus on basic structural and semantic relationships without the computational overhead of detailed features. 
In the second stage, we introduce medium-resolution inputs, expanding the model’s capacity to capture more nuanced patterns while gradually increasing sequence length.
Finally, in the high-resolution stage, the model is trained on longer sequences with a broader range of input variability, enhancing its ability to generalize to diverse, complex visual and linguistic tasks. 
This staged approach ensures a smooth transition from coarse to fine-grained learning, while maintaining models' capabilities.


| -------- | ------- |
| GPUs  | 128 H100-80G   |
| Training time | 14 Days     |
| Training data	   | Cambrian-7M    |


## Acknowledgment
This project is built upon the following awesome projects [LLaVA](https://github.com/haotian-liu/LLaVA), [Open-LLaVA-NeXT](https://github.com/xiaoachen98/Open-LLaVA-NeXT).
We thank AI21 Labs and Technology Innovation Institute for open-sourcing the powerful LLMs.
We also thank the [Cambrian-1](https://cambrian-mllm.github.io/) project for providing such high-quality vision-language datasets.

## Citation

The paper is coming soon. Meanwhile, please use the following to cite:
```
@article{vipervlm,
  title={Viper: Open Mamba-based Vision-Language Models},
  author={Zhuang, Yufan and Chuang, Pierce and Lu, Yichao and Harpale, Abhay and Bhardwaj, Vikas and Shang, Jingbo},
  year={2024}
}
```
