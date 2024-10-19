from openai import OpenAI
from pydantic import BaseModel

class HP(BaseModel):
    reason: str
    batch_size: int
    lr: float
    num_epochs: int
    num_workers: int
    decoder: str
    num_decoder_layers: int
    image_size: int
    dropout: float
    ffn_dim: int
    last_n: int
    backbone: str
    frozen_backbone: bool
    weight_decay: float
    loss_fn: str
    use_bsvunet2_style_dataaug: bool
    conf_pen: float
    base_aug_spice: float
    use_train_as_val: bool
    parallel_bca: int 
    fusion: str
    cross_attention: str


client = OpenAI(api_key="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")

prompt = """
You are now an intelligent hyperparameter optimization agent for deep learning models. You will receive the model description and the current set of hyperparameters. Based on this information, your task is to suggest the next iteration of hyperparameters. Afterward, you will receive updated performance metrics (e.g., validation error, validation loss, training loss, validation IOU, etc.). Use these metrics to adjust the hyperparameters for the next iteration to improve model performance. State your reasoning. Response your suggestions in the JSON format, only json and nothing else. Give through reasoning before start


Model Description:

You are tuning a background subtraction model with the following architecture:

Backbone: You have the choice between two backbone models, Segformer or DYNO.
Cross-attention: This mechanism compares the current frame with a long background frame and a short background frame, identifying differences between them.
Decoder: After attention, the data is passed to a decoder, which could be either a Convolutional Neural Network (CNN) or a Transformer without positional embeddings.
Output Layer: A small CNN is used for the output.
To begin, please provide:

The current hyperparameter settings for the backbone, cross-attention, decoder, and the small CNN.
The metrics from the last iteration (such as validation error, validation loss, IOU, etc.).
I'll use this information to suggest the next iteration of hyperparameters.



Hyperparameters:
    batch_size 
    lr: learning rate
    num_epochs: num of epochs
    num_workers: keep it 50
    decoder: decoder, coule be transformer or conv
    num_decoder_layers: number of decoder layers in the model, only used if decoder is transforme
    image_size: image size of the side
    dropout: dropout rate
    ffn_dim: feed forward dimension
    last_n; how many layes of the backbone to use feeding to the attention, only used if backbone is dino
    backbone: could only be segformer for now 
    frozen_backbone: whether to freeze the backbone, True or False
    weight_decay: weight decay
    loss_fn: loss function, could be iou or f1
    use_bsvunet2_style_dataaug: if true, use the data augmentation style of bsvunet2
    conf_pen: confidence penalty to penalize the model for being too confident by adding torch.mean((pred - 0.5)**2) * args.conf_pen to the loss
    base_aug_spice: how spicy the base augmentations are (0-1)
    use_train_as_val: use the training set as the validation set, for debugging purposes
    parallel_bca: how many parallel cross attention heads to use 
    fusion: how to fuse the current frame with the background, could be concat or cross_attention
    cross_attention: only if fusion is cross_attention, how to fuse the current frame with the background, could be lite or full
Now you start with: (start your first try with these hyperparameters and do not try to think or adjust or "improve"! Just start with these hyperparameters and see how it goes, for first round, say "use pre set" as reasoning)
'batch_size': 32, 'lr': 6e-06, 'num_epochs': 700, 'num_workers': 20, 'decoder': 'transformer', 'num_decoder_layers': 2, 'image_size': 384, 'dropout': 0.18, 'ffn_dim': 1024, 'last_n': 1, 'backbone': 'segformer', 'frozen_backbone': False, 'weight_decay': 2.5e-06, 'loss_fn': 'f1', 'use_bsvunet2_style_dataaug': True, 'conf_pen': 0.03, 'base_aug_spice': 0.4, 'use_train_as_val': False, 'parallel_bca': 4, 'fusion': 'concat', cross_attention: lite

If you encounter keyboard interrupt, that means the user think the current run is not going well and want to stop the current run and start a new one. 
"""




import json

messages = [{"role": "user","content": prompt}]
import os
for i in range(30):
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format = HP
    )
    response = json.loads(response.choices[0].message.content)
    params = ""
    for key, value in response.items():
        if key == "reason":
            continue
        if type(value) != bool:
            params += f"--{key} {value} "
        elif value:
            params += f"--{key} "
    print(response)
    commond_line = f"python3 main.py {params}--log_file {i}.log 2> error.log"
    print(commond_line)
    messages.append({"role": "assistant", "content": json.dumps(response)})
    os.system("echo > error.log")
    os.system(f"echo > {i}.log")
    os.system(commond_line)
    with open("error.log") as f:
        error = f.read()
    with open(f"{i}.log") as f: 
        log = f.read()

    msg = f"Error:\n{error}\n\nLog:\n{log}"
    messages.append({"role": "user", "content": msg})
    with open("chat.json", "w") as f:
        f.write(json.dumps(messages, indent=4)) 
