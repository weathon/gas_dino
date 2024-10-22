from openai import OpenAI
from pydantic import BaseModel

class HP(BaseModel):
    reason: str
    batch_size:int
    lr:float
    num_epochs:int
    decoder: str
    image_size: int
    dropout: float
    backbone: str
    weight_decay: float
    loss_fn: str
    use_bsvunet2_style_dataaug: bool
    conf_pen: float
    base_aug_spice:float
    fusion: str
    gpu: str
        


client = OpenAI()

prompt = """
You are now an intelligent hyperparameter optimization agent for deep learning models. You will receive the model description and the current set of hyperparameters. Based on this information, your task is to suggest the next iteration of hyperparameters. Afterward, you will receive updated performance metrics (e.g., validation error, validation loss, training loss, validation IOU, etc.). Use these metrics to adjust the hyperparameters for the next iteration to improve model performance. State your reasoning. Response your suggestions in the JSON format, only json and nothing else. Give through reasoning before start


Model Description:

You are tuning a background subtraction model with the following architecture:
The model is a segmentation model. With segformer as backbone and a conv as decoder. 


Hyperparameters:
batch_size keep it small around 8 
lr
num_epochs
decoder: keep it none
image_size: side length of the image, could be 224-512 or higher if needed
dropout: drop out rate
backbone: keep it as segformer
weight_decay: weight decay, around 0.01 to begin
loss_fn: f1 or iou, keep it as f1
use_bsvunet2_style_dataaug: if True, use bsvunet2 style data augmentation
conf_pen: penlty for being too confident
base_aug_spice: how spicy is the base augmentation
fusion: keep it as early
gpu: keep it as 1

For the first round, use the following hyperparameters and only put "as is" in the reason. Ignore the parameters not listed above. 
main.py --batch_size 16 --lr 5e-06 --num_epochs 800 --num_workers 30 --decoder none --num_decoder_layers 2 --image_size 512 --dropout 0.7 --ffn_dim 1024 --last_n 1 --backbone segformer --weight_decay 0.02 --loss_fn f1 --use_bsvunet2_style_dataaug --conf_pen 0.001 --base_aug_spice 0.7 --parallel_bca 4 --fusion early --cross_attention lite --use_tqdm --gpu 1
If it dose not converges well, try decrease the batch size to 8 


User hint: if the f1 did not increase at the END, it could because the learning rate is too low at the end. If it did not increase at the BEGINNING,
 it could be because the learning rate is too high OR too low. Note that at the end it could increase slowly and with noise. Do not treat this as no
 increase and might increase epochs. Also, if the program exited with keyboard interp error, that means user discarded the suggestion,
 use your judgement to decide if you want to continue with the same suggestion or not.
"""




import json

messages = [{"role": "user","content": prompt}]
import os
for i in range(30):
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        response_format = HP,
        store = True
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
    commond_line = f"python3 main.py {params}--use_optical_flow --log_file {i}_op.log 2> error_op.log"
    print(commond_line)
    messages.append({"role": "assistant", "content": json.dumps(response)})
    os.system("echo > error_op.log")
    os.system(f"echo > {i}_op.log")
    os.system(commond_line)
    with open("error_op.log") as f:
        error = f.read()
    with open(f"{i}_op.log") as f: 
        log = f.read()

    msg = f"Error:\n{error}\n\nLog:\n{log}"
    messages.append({"role": "user", "content": msg})
    with open("chat_op.json", "w") as f:
        f.write(json.dumps(messages, indent=4)) 
