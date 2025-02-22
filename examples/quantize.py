import sys
import os
sys.path.append(os.path.abspath("/workspace/AutoAWQ"))

print(sys.path)

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch


num_gpus = torch.cuda.device_count()
print(f"系统中可用的 GPU 数量: {num_gpus}")

# 遍历所有 GPU 并打印名称
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
model_path = 'Qwen/Qwen2-7B-Instruct'
quant_path = 'Qwen2-7B-Instruct-awq-modified'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
print(model)
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')