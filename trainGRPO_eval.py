import os
from unsloth import FastLanguageModel
import torch
import json
import re
from datasets import load_dataset, Dataset, load_from_disk
from sklearn.model_selection import train_test_split
import tensorboard
from unsloth import FastLanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset
import re
from tqdm import tqdm
import torch
from my_grpo_trainer import MyGRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from trl import SFTTrainer


# 固定随机种子
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)


max_seq_length = 1024 # Can increase for longer reasoning traces
max_prompt_length = 256
lora_rank = 64 # Larger rank = smarter, but slower # 数学推理任务最好设置高一点
output_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/nitingjuntao-240104020001/RLhw/outputs_rft6"
logging_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/nitingjuntao-240104020001/RLhw/logs"
log_path = os.path.join(output_dir, "eval_results.log")
max_steps = 250


model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tongliwen-240107020010/Project/LLMRFT/models/meta-llama3.1-8B-instruct",
    model_name = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/nitingjuntao-240104020001/RLhw/model/LLM-Research/Meta-Llama-3-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit # 4bit 量化，LoRA 微调用这个精度是可以的
    fast_inference = False, # Enable vLLM fast inference # 推理加速，训练的时候不要开
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9, # Reduce if out of memory # 对于单卡单任务，这个显存使用上限可以调高一点，调到 1 都行
)
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank * 2, # 推荐设置为 2r
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

# 指定本地路径保存数据集
dataset = load_from_disk(
    "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/nitingjuntao-240104020001/RLhw/dataset/gsm8k"
)

# 系统提示（推理格式）  -  补全换行符
SYSTEM_PROMPT = (
    "Respond in the following format:\n"
    "<reasoning>\n"
    "...reasoning steps...\n"
    "</reasoning>\n"
    "<answer>\n"
    "...final answer...\n"
    "</answer>\n"
)

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def process_data(data):
    response_text = data['answer']
    reasoning = response_text.split('####')[0].strip() # 提取推理过程
    answer = response_text.split('####')[-1].strip() # 提取答案部分
    formatted_answer = XML_COT_FORMAT.format(reasoning=reasoning, answer=answer)
    # if formatted_answer is None:
    #     return None
    return {
        'question': data['question'],
        'response': formatted_answer,
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': data['question']}
        ],
        'answer': answer
    }

filtered_dataset = dataset.filter(lambda x: '####' in x['answer'])
formatted_data = filtered_dataset.map(process_data)
train_data = formatted_data['train']
test_data = formatted_data['test']
    
def prepare_for_rft_format(example, tokenizer):
    # 使用 tokenizer 生成 LLaMA3 格式的 prompt（包含 assistant 标头提示）
    prompt_text = tokenizer.apply_chat_template(
        example["prompt"],
        tokenize=False,
        add_generation_prompt=False
    )

    return {
        "input": prompt_text.strip(),            # 模型实际使用的 prompt（带 assistant 起始）
        "answer": example["answer"].strip(),     # gold 数值答案
        # "response": example["response"].strip()  # 可留可不留
    }
    
    
def formatting_prompts_func(example):
    messages = example["prompt"]

    # 构造 system + user 段落（不含 assistant prompt）
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False  # SFT 不能让 assistant 起头
    )

    # 拼接 assistant 段头 + response + LLaMA3 专用结束符
    text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    text += example["response"].strip() + "<|eot_id|>"

    return text

# 应用格式转换
train_data = train_data.map(lambda x: {"text": formatting_prompts_func(x)})
test_data = test_data.map(lambda x: {"text": formatting_prompts_func(x)})

# 过滤空数据
train_data = train_data.filter(lambda x: len(x["text"]) > 0)
test_data = test_data.filter(lambda x: len(x["text"]) > 0)



# # 计算划分数量（5%）
# train_list = train_data.to_list()
# train_part, val_part = train_test_split(train_list, test_size=0.05, random_state=42)

# # 重新转换为 Dataset 格式
# train_data = Dataset.from_list(train_part)
# val_data = Dataset.from_list(val_part)

# 精度评估
def extract_all_numbers(text: str) -> list[str]:
    """提取所有可能的数字字符串（支持 $, , 分隔）"""
    pattern = r"[-+]?\$?\d[\d,]*\.?\d*"
    matches = re.findall(pattern, text)
    cleaned = [m.replace(",", "").replace("$", "") for m in matches]
    return cleaned

def extract_assistant_response(full_output: str) -> str:
    """仅提取 assistant 的回答部分"""
    match = re.search(r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*)", full_output, re.DOTALL)
    return match.group(1) if match else full_output

def evaluate_single_response_final(full_output: str, gold_answer: str) -> dict:
    gold_answer = gold_answer.strip()
    
    # 提取 assistant 回答部分
    assistant_output = extract_assistant_response(full_output)
    
    # 清除 <|xxx|> 控制符：容忍控制符的存在，除此之外的多余字符算格式错误
    cleaned = re.sub(r"<\|.*?\|>", "", assistant_output).strip()

    format_correct = False
    answer_correct = False
    strict_answer_correct = False

    pattern = re.compile(
        r"^<reasoning>\n(.+?)\n</reasoning>\n<answer>\n(.+?)\n</answer>\s*$",
        flags=re.DOTALL
    )

    match = pattern.match(cleaned)
    if match:
        format_correct = True
        answer_block = match.group(2)
        answer_nums = extract_all_numbers(answer_block)
        if answer_nums and answer_nums[0] == gold_answer: # 要求答案出现在首位
            strict_answer_correct = True

    all_nums = extract_all_numbers(cleaned)
    if gold_answer in all_nums: # 只要出现答案就算对
        answer_correct = True

    return {
        "format_correct": format_correct,
        "answer_correct": answer_correct,
        "strict_answer_correct": strict_answer_correct
    }
    


def collate_fn_llama3(batch, tokenizer, max_length=2048):
    prompts = [ex["prompt"] for ex in batch]
    golds = [str(ex["answer"]).strip() for ex in batch]
    questions = [ex["question"] for ex in batch]

    # 预处理为纯字符串，避免 tokenizer 内部 bug
    prompt_texts = [
        tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True
        ) for msg in prompts
    ]

    # 编码前确保 padding 设置生效
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    return encoded["input_ids"], encoded["attention_mask"], golds, questions


def evaluate_dp(
    model,
    tokenizer,
    dataset,
    max_samples: int = 100,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    record_errors: bool = False
):
    # ✅ 设置 padding_side & pad_token（仅作为冗余保险）
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    total = 0
    format_correct = 0
    answer_correct = 0
    strict_correct = 0
    format_errors, answer_errors, strict_errors = [], [], []

    dataloader = DataLoader(
        dataset.select(range(max_samples)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn_llama3(x, tokenizer),
    )

    for input_ids, attention_mask, golds, questions in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

        for i in range(len(decoded_outputs)):
            result = evaluate_single_response_final(decoded_outputs[i], golds[i])
            total += 1

            if result["format_correct"]:
                format_correct += 1
            elif record_errors:
                format_errors.append((questions[i], decoded_outputs[i]))

            if result["answer_correct"]:
                answer_correct += 1
            elif record_errors:
                answer_errors.append((questions[i], decoded_outputs[i]))

            if result["strict_answer_correct"]:
                strict_correct += 1
            elif record_errors:
                strict_errors.append((questions[i], decoded_outputs[i]))

    format_acc = format_correct / total
    answer_acc = answer_correct / total
    strict_acc = strict_correct / total

    print(f"\n📐 Format Accuracy: {format_acc:.2%} ({format_correct}/{total})")
    print(f"🔢 Answer Accuracy: {answer_acc:.2%} ({answer_correct}/{total})")
    print(f"🎯 Strict Accuracy: {strict_acc:.2%} ({strict_correct}/{total})")

    return {
        "format_accuracy": format_acc,
        "answer_accuracy": answer_acc,
        "strict_accuracy": strict_acc,
        "format_errors": format_errors if record_errors else None,
        "answer_errors": answer_errors if record_errors else None,
        "strict_errors": strict_errors if record_errors else None,
    }
    
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print(f"Padding side: {tokenizer.padding_side}")
print(f"Pad token: {tokenizer.pad_token} / ID: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token} / ID: {tokenizer.eos_token_id}")

# tokenizer.padding_side = "right"
# tokenizer.pad_token = tokenizer.eos_token

# print(f"Padding side: {tokenizer.padding_side}")
# print(f"Pad token: {tokenizer.pad_token} / ID: {tokenizer.pad_token_id}")
# print(f"EOS token: {tokenizer.eos_token} / ID: {tokenizer.eos_token_id}")

def format_reward_func(completions, **kwargs) -> list[float]:
    rewards = []
    for c in completions:
        text = extract_assistant_response(c[0]["content"])
        cleaned = re.sub(r"<\|.*?\|>", "", text).strip() # 去除 EOS 标识符

        pattern = re.compile(
            r"^<reasoning>\n(.+?)\n</reasoning>\n<answer>\n(.+?)\n</answer>\s*$",
            flags=re.DOTALL
        )
        match = pattern.match(cleaned)
        rewards.append(1.0 if match else 0.0)
    return rewards

def answer_inclusion_reward_func(completions, answer, **kwargs) -> list[float]:
    gold = str(answer[0]).strip()
    rewards = []
    for c in completions:
        text = extract_assistant_response(c[0]["content"])
        cleaned = re.sub(r"<\|.*?\|>", "", text).strip()
        nums = extract_all_numbers(cleaned)
        rewards.append(1.0 if gold in nums else 0.0)
    return rewards

def strict_answer_reward_func(completions, answer, **kwargs) -> list[float]:
    gold = str(answer[0]).strip()
    rewards = []
    for c in completions:
        text = extract_assistant_response(c[0]["content"])
        cleaned = re.sub(r"<\|.*?\|>", "", text).strip()

        pattern = re.compile(
            r"^<reasoning>\n(.+?)\n</reasoning>\n<answer>\n(.+?)\n</answer>\s*$",
            flags=re.DOTALL
        )
        match = pattern.match(cleaned)
        if match:
            answer_block = match.group(2)
            answer_nums = extract_all_numbers(answer_block)
            if answer_nums and answer_nums[0] == gold:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

# 用于测试 trainer 的行为
def debug_reward(prompts, completions, answer, **kwargs):
    print("💬 PROMPT:", prompts[0][-1]["content"])
    print("🧾 COMPLETION:", completions[0][0]["content"])
    print("🎯 GOLD ANSWER:", answer[0])
    return [1.0]  # dummy score





# = TrainingArguments(
#     per_device_train_batch_size=72,          # GRPO 每步训练样本数 = 12 × 6 → SFT 一步训练72条等效
#     gradient_accumulation_steps=1,           # 与 GRPO 保持一致

#     learning_rate=5e-6,                      # 对齐
#     adam_beta1=0.9,
#     adam_beta2=0.99,
#     weight_decay=0.1,
#     warmup_ratio=0.1,
#     lr_scheduler_type="cosine",              # 对齐 GRPO
#     optim="paged_adamw_8bit",                # 和 Unsloth 版本完全一致

#     max_steps=500,                           # 与 GRPO 一致
#     max_grad_norm=0.1,                       # GRPO 是 0.1，SFT 也同步

#     save_strategy="steps",
#     save_steps=250,
#     save_total_limit=2,

#     logging_steps=1,
      
#     fp16=not torch.cuda.is_bf16_supported(),
#     bf16=torch.cuda.is_bf16_supported(),

#     report_to=["tensorboard"],
#     logging_dir=logging_dir,
#     output_dir=output_dir,
# )


# # 创建训练器
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=train_data,
#     # eval_dataset=test_data,
#     dataset_text_field="text",
#     max_seq_length=max_seq_length,
#     args=training_args,
#     packing=False,  # 数学推理任务建议关闭packing
# )

# # 开始训练
# trainer.train()


tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print(f"Padding side: {tokenizer.padding_side}")
print(f"Pad token: {tokenizer.pad_token} / ID: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token} / ID: {tokenizer.eos_token_id}")

# rft
# 应用格式转换
train_data = train_data.map(lambda x: prepare_for_rft_format(x, tokenizer))
test_data = test_data.map(lambda x: prepare_for_rft_format(x, tokenizer))


# for n, p in model.named_parameters():
#     if "lora_" not in n:
#         p.requires_grad = False
        
        
# fmt_tokens = [
#     ".", ",", ":", ";", "!", "?", "\n", "<newline>", "(", ")", "[", "]"
# ]
# format_token_ids = {
#     tokenizer.convert_tokens_to_ids(t)
#     for t in fmt_tokens
#     if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id
# }


from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 12,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = max_steps, 
    save_steps = 250, 
    max_grad_norm = 0.1,
    report_to = ["tensorboard"], # Can use Weights & Biases # Weights & Biases 需要联网，这里选择用 Tensorboard 
    logging_dir = logging_dir,
    output_dir = output_dir,
)



trainer = MyGRPOTrainer(
    model = model,
    # processing_class = tokenizer,
    tokenizer = tokenizer,
    reward_funcs = [
        # debug_reward,
        format_reward_func,
        answer_inclusion_reward_func,
        strict_answer_reward_func
    ],
    args = training_args,
    train_dataset = train_data,
    # beta_base=0.01,
    # beta_fmt=0.10,
    # format_token_ids=format_token_ids,
)

trainer.train()



tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print(f"Padding side: {tokenizer.padding_side}")
print(f"Pad token: {tokenizer.pad_token} / ID: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token} / ID: {tokenizer.eos_token_id}")



# evaluation


ckpts = []
for name in os.listdir(output_dir):
    m = re.match(r"checkpoint-(\d+)", name)
    if m:
        step = int(m.group(1))
        ckpts.append((step, os.path.join(output_dir, name)))
ckpts.sort(key=lambda x: x[0])

# 2) 打开日志文件，写入表头
best_acc = -1.0
best_ckpt = None

accuracies = {}

with open(log_path, "w") as log_file:
    log_file.write("step\tstrict_acc\tanswer_acc\tformat_acc\n")

    for step, ckpt_dir in ckpts:
        print(f"\n▶ Evaluating checkpoint {step} …")
        try:
            # 3) 从 checkpoint 中加载模型
            # model_ckpt = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map="auto")
            model_ckpt, tokenizer_ckpt = FastLanguageModel.from_pretrained(
                model_name     = ckpt_dir,
                max_seq_length = max_seq_length - max_prompt_length,
                load_in_4bit   = True,
                fast_inference = False,
                device_map     = "auto",
            )
            
            # 4) 调用你的 evaluate_accuracy 函数
            metrics = evaluate_dp(
                model=model_ckpt,
                tokenizer=tokenizer_ckpt,
                dataset=test_data,
                max_samples=len(test_data),
                batch_size=128,
                max_new_tokens=max_seq_length - max_prompt_length,
                record_errors=False
            )
            acc = metrics["strict_accuracy"]
             
        except Exception as e:
            print(f"Failed to evaluate checkpoint {step}: {e}")
            continue

        # 5) 记录并写日志
        accuracies[step] = acc
        # log_file.write(f"{step}\t{acc:.4f}\n")
        log_file.write(
            f"{step}\t{metrics['strict_accuracy']:.4f}"
            f"\t{metrics['answer_accuracy']:.4f}"
            f"\t{metrics['format_accuracy']:.4f}\n")
        log_file.flush()
        print(f"Accuracy = {acc:.4%}")

