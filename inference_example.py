"""Run local inference with the custom, randomly-initialized-then-trained model.

Usage: python inference_example.py artifacts/airquality-miniqwen-scratch "Classify ..."
"""
import os
os.environ['USE_TF'] = '0'
import sys, json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from airquality_slm_model import MiniQwenModel, MiniQwenConfig


def main():
    folder=Path(sys.argv[1]) if len(sys.argv)>1 else Path('artifacts/airquality-miniqwen-scratch')
    instruction=(sys.argv[2] if len(sys.argv)>2 else 'Classify these historical air-quality measurements as low, typical, elevated or unavailable relative to training-distribution quartiles. Hour: 2004-03-10T18:00:00. CO: 2.60 mg/m3; NO2: 113.00 ug/m3; benzene: 11.90 ug/m3.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config=MiniQwenConfig(**json.loads((folder/'model_config.json').read_text(encoding='utf-8')))
    model=MiniQwenModel(config)
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
    model.to(device).eval()
    tokenizer=AutoTokenizer.from_pretrained(str(folder))
    prefix=f'Instruction:\n{instruction}\nAnswer:\n'
    ids=tokenizer.encode(prefix,add_special_tokens=False)[-(config.max_seq_len-50):]
    input_ids=torch.tensor([ids],dtype=torch.long,device=device)
    result=model.generate(input_ids,eos_token_id=tokenizer.eos_token_id,max_new_tokens=50)
    print(tokenizer.decode(result[0,input_ids.shape[1]:].tolist(),skip_special_tokens=True))

if __name__=='__main__':
    main()
