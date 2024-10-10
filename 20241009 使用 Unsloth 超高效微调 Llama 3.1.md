## 使用 Unsloth 超高效微调 Llama 3.1
Llama 3.1 的发布带来了极高的性能表现，缩小了开源模型与闭源模型的差距。相比直接使用冻结的通用大语言模型 (LLM)，例如 GPT-4o 和 Claude 3.5，您可以选择微调 Llama 3.1，针对特定场景进行优化，从而在成本较低的情况下实现更好的性能和定制化。



本文将系统介绍监督微调 (Supervised Fine-Tuning, SFT) 的相关知识，探讨其与提示工程 (prompt engineering) 的对比，说明在何种场景下更适合使用微调。同时，我们将详细介绍主要技术的优缺点，并引入一些核心概念，如 LoRA 超参数、存储格式和聊天模板。最后，我们将在 Google Colab 上实际操作，利用 Unsloth 对 Llama 3.1 8B 进行微调，并采用当前最先进的优化方法。

### 监督微调
监督微调 (SFT) 是一种提升和定制预训练大语言模型的方法。它通过在较小的数据集上进行再训练，将基础模型转化为能够执行指令和回答问题的智能助手。同时，SFT 还能提升模型整体性能、增加新知识或适应特定任务和领域。微调后的模型还可以通过偏好对齐进一步优化（详细内容参见 我的 DPO 文章），以移除不必要的回复或调整风格。

下图展示了一个典型指令示例，包括一个用于引导模型的系统提示、一个提供任务的用户提示，以及模型预期生成的输出。高质量的开源指令数据集可以在 💾 LLM Datasets GitHub 仓库中找到。



在考虑 SFT 之前，建议优先尝试少样本提示或检索增强生成 (retrieval augmented generation, RAG) 等提示工程技术。在实践中，这些方法通常能有效解决大多数问题，无需微调，适用于闭源或开源权重的模型 (如 Llama 3.1 Instruct)。如果提示工程在质量、成本或延迟等方面不能满足需求，且有可用的指令数据，那么 SFT 是一种可行的选择。SFT 还能提供额外的控制和定制化，有助于打造个性化的大语言模型。

但 SFT 也有局限性。它在利用基础模型已有知识时效果最好，而学习全新信息（如一种未知语言）会更加困难，容易导致幻觉现象。如果是基础模型尚未知晓的新领域，建议先在原始数据集上进行持续预训练。

另外，已经经过微调的指令模型 (instruct model) 通常已经接近您的需求。例如，一个模型可能在性能上表现良好，但会表明它由 OpenAI 或 Meta 训练而非您。在这种情况下，您可以通过偏好对齐稍微调整其行为。通过提供 100 到 1000 个指令样本的选择和拒绝案例，可以让 LLM 声称是由您训练，而非 OpenAI。

### 微调 Llama 3.1 8B
为了高效地微调 Llama 3.1 8B 模型，我们将使用由 Daniel 和 Michael Han 开发的 Unsloth 库。得益于其定制内核，Unsloth 提供了 2 倍的训练速度和 60% 的内存使用率，是在 Colab 这样的受限环境中理想的选择。目前 Unsloth 仅支持单 GPU 设置。对于多 GPU 配置，建议使用 TRL 和 Axolotl 等流行方案（它们也使用 Unsloth 作为后端）。

在此示例中，我们将使用 QLoRA 在 mlabonne/FineTome-100k 数据集上微调。该数据集是 arcee-ai/The-Tome 的一个子集（不包括 arcee-ai/qwen2-72b-magpie-en），经过我使用 HuggingFaceFW/fineweb-edu-classifier 重新筛选。虽然该分类器并非专门用于指令数据质量评估，但可以作为粗略参考。最终得到的 FineTome 是一个高质量数据集，涵盖对话、推理、函数调用等多种任务。

让我们先安装所有必要的库。

~~~
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
~~~

安装完成后，我们按如下方式导入这些库。

~~~python
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
~~~
现在让我们加载模型。由于我们打算使用 QLoRA，因此选择了预量化的 unsloth/Meta-Llama-3.1-8B-bnb-4bit。这个 4 位精度版本的 meta-llama/Meta-Llama-3.1-8B 模型比原始的 16 位精度模型 (16 GB) 要小得多 (5.4 GB)，因此下载速度更快。我们使用 bitsandbytes 库以 NF4 格式加载模型。

在加载模型时，需要指定一个最大序列长度参数，它决定了模型的上下文窗口大小。Llama 3.1 支持最多 128k 的上下文长度，但在这个例子中我们将其设置为 2,048，因为更长的上下文会增加计算和显存的消耗。最后，dtype 参数可以自动检测你的 GPU 是否支持 BF16 格式，这种格式能在训练中提高稳定性 (仅限于 Ampere 或更新的 GPU)。

~~~python
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)
~~~
现在模型已加载为 4 位精度，我们将配置 LoRA 适配器以实现参数高效微调。LoRA 的三个关键参数如下：

- Rank (r)：决定 LoRA 矩阵的大小。Rank 通常从 8 开始，但可以达到 256。更高的 Rank 可以容纳更多信息，但同时也增加了 LoRA 的计算和内存开销。这里我们将其设置为 16。
- Alpha (α)：更新时的缩放因子。Alpha 直接影响适配器的作用，通常设为 Rank 值的 1 倍或 2 倍。
- 目标模块：LoRA 可应用于模型的多个组件，包括注意力机制 (Q、K、V 矩阵)、输出投影、前馈块和线性输出层。虽然最初主要针对注意力机制，但将 LoRA 扩展到其他模块也显示出潜力。不过，适配更多模块会增加可训练参数和内存需求。
- 在这里，我们将 r=16、α=16，并目标指向所有线性模块以确保高质量输出。我们不使用 dropout 和偏置项以提高训练速度。

此外，我们还将采用 Rank-Stabilized LoRA (rsLoRA)，它将 LoRA 适配器的缩放因子设为 1/√r，这比传统的 1/r 更有利于稳定学习 (尤其是对于更高 Rank 的适配器)，从而提升微调效果。Unsloth 通过梯度检查点管理，将输入和输出嵌入转移到磁盘以节省显存。

~~~python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)
~~~

通过这种 LoRA 配置，我们只训练了 8 亿个参数中的 42 百万个 (占比 0.5196%)。这表明，与全量微调相比，LoRA 高效得多。

现在我们加载并准备数据集。指令数据集通常以一种特定格式存储，例如 Alpaca、ShareGPT 或 OpenAI 格式。首先，我们需要解析这些格式以提取指令和答案。我们的 mlabonne/FineTome-100k 数据集采用了 ShareGPT 格式，其中包含一个 JSONL 格式的“conversations”列记录对话内容。与简单的 Alpaca 格式不同，ShareGPT 更适合存储多轮对话，这更接近用户与大语言模型的交互方式。

解析出指令-回答对后，我们会重新格式化它们以匹配一种聊天模板。聊天模板是一种结构化用户与模型对话的方式，通常包含标识消息起止点及说话者身份的特殊 Token。基础模型并没有内置聊天模板，因此我们可以根据需要选择 ChatML、Llama3 或 Mistral 等。在开源社区中，ChatML 模板 (最初由 OpenAI 推出) 是一个广泛使用的选项。它只需添加两个特殊 Token (<|im_start|> 和 <|im_end|>) 来标识对话的发言者。

如果我们将这个模板应用到之前的指令示例，结果如下：

~~~xml
<|im_start|>system
You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.<|im_end|>
<|im_start|>user
Remove the spaces from the following sentence: It prevents users to suspect that there are some hidden products installed on theirs device.
<|im_end|>
<|im_start|>assistant
Itpreventsuserstosuspectthattherearesomehiddenproductsinstalledontheirsdevice.<|im_end|>
~~~

在接下来的代码块中，我们通过 mapping 参数解析 ShareGPT 数据集并包含 ChatML 模板。然后，我们加载并处理整个数据集，将聊天模板应用到每个对话中。

~~~python
tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = dataset.map(apply_template, batched=True)
~~~

现在我们可以开始配置训练参数。我将简要介绍一些关键超参数：

- 学习率：它决定了模型更新参数的力度。如果学习率太低，训练进展会很慢，且可能卡在局部最优解；而如果学习率过高，训练可能变得不稳定甚至发散，从而降低模型性能。
- 学习率调度器：它会在训练过程中动态调整学习率，通常在初期使用较高学习率以快速进展，随后逐步降低。线性和余弦调度是最常见的两种方案。
- 批次大小：每次权重更新前处理的样本数量。更大的批次通常能带来更稳定的梯度估计并提高训练速度，但也需要更多内存。通过梯度累积，可以在多个前向/后向传递中累积梯度，以达到更大的有效批次大小。
- 训练轮数：模型遍历整个训练集的次数。更多的训练轮数可以让模型更充分地学习数据，可能带来更好的性能。然而，训练轮数过多可能会导致过拟合。
- 优化器：用于调整模型参数以最小化损失函数的算法。通常建议使用 8 位的 AdamW：它在内存占用更少的情况下与 32 位版本表现相当。AdamW 的分页版本仅在分布式训练环境中有意义。
- 权重衰减：一种正则化技术，它在损失函数中添加大权重的惩罚，从而防止过拟合，帮助模型学习更简单、更具泛化能力的特征。但权重衰减过大可能抑制模型学习。
- 预热步数：在训练初期，学习率从较低值逐步增加到初始设定值的过程。预热有助于稳定早期训练，尤其是在使用较大学习率或批次时，通过让模型在进行大幅度更新之前逐步适应数据分布。
- 数据打包：批次中有预定义的序列长度。我们可以将多个较小的样本打包成一个批次，以提高效率。
我在 Google Colab 上使用 A100 GPU (40 GB 显存) 训练了整个数据集 (100k 样本)。训练耗时 4 小时 45 分钟。当然，你也可以使用显存较小的 GPU 和较小的批次大小，但速度会慢得多。例如，在 L4 上大约需要 19 小时 40 分钟，而在免费的 T4 上则需要长达 47 小时。

在这种情况下，我建议只加载数据集的一部分以加快训练。你可以通过调整前面代码块实现，比如 dataset = load_dataset("mlabonne/FineTome-100k", split="train[:10000]") 只加载 10k 样本。或者，你可以选择更便宜的云 GPU 提供商，如 Paperspace、RunPod 或 Lambda Labs。

~~~
trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

trainer.train()
~~~

模型训练完成后，让我们通过一个简单的提示进行测试。这虽然不是严格的评估，但可以快速检查潜在问题。我们使用 FastLanguageModel.for_inference() 来实现 2 倍速的推理。

~~~
model = FastLanguageModel.for_inference(model)

messages = [
    {"from": "human", "value": "Is 9.11 larger than 9.9?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)
~~~

模型的输出是 "9.9"，这是正确的！

现在让我们保存训练好的模型。如果你还记得 LoRA 和 QLoRA 的部分，我们其实并没有训练整个模型，而是一个适配器模块。在 Unsloth 中，有三种保存方式：lora 仅保存适配器，merged_16bit 和 merged_4bit 则是在 16 位或 4 位精度下将适配器与模型合并保存。

接下来，我们选择以 16 位精度合并，以确保最佳质量。我们先将模型保存到本地的 "model" 目录，然后上传到 Hugging Face Hub。你可以在 mlabonne/FineLlama-3.1-8B 找到这个训练好的模型。

```python
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged("mlabonne/FineLlama-3.1-8B", tokenizer, save_method="merged_16bit")
```

Unsloth 还提供了将模型直接转换为 GGUF 格式的功能。这是一种为 llama.cpp 设计的量化格式，兼容大多数推理引擎，比如 LM Studio、Ollama 和 oobabooga 的 text-generation-webui。你可以根据需要选择不同的量化精度 (详见 我关于 GGUF 和 llama.cpp 的文章)，我们会遍历一个列表，依次以 q2_k、q3_k_m、q4_k_m、q5_k_m、q6_k 和 q8_0 进行量化，并将这些量化版本上传到 Hugging Face。你可以在 mlabonne/FineLlama-3.1-8B-GGUF 找到所有这些 GGUF 文件。

~~~python
quant_methods = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0"]
for quant in quant_methods:
    model.push_to_hub_gguf("mlabonne/FineLlama-3.1-8B-GGUF", tokenizer, quant)
~~~

恭喜你，我们从头开始微调了一个模型，并上传了不同量化版本，你现在可以在喜欢的推理引擎中使用它们。可以随时试试最终的模型版本，它已经在 mlabonne/FineLlama-3.1-8B-GGUF 上架。接下来可以做什么？以下是一些建议：

- 评估：可以在 Open LLM Leaderboard 上进行评估 (免费提交)，或者使用其他评估工具，比如 LLM AutoEval。
- 对齐：可以使用偏好数据集，如 mlabonne/orpo-dpo-mix-40k，通过直接偏好优化 (Direct Preference Optimization) 来提升模型表现。
- 量化：使用 AutoQuant 将模型转换为其他格式，比如 EXL2、AWQ、GPTQ 或 HQQ，以实现更快的推理或更低精度。
- 部署：如果模型经过足够的训练 (约 20k 样本)，可以使用 ZeroChat 在 Hugging Face Space 上进行部署。

### 结论
本文详细介绍了如何对 Llama 3.1 8B 模型进行监督微调，并提供了实操步骤。通过利用 QLoRA 的高效内存使用，我们成功在有限的 GPU 资源下微调了一个 8B 大语言模型，并提供了适合更大规模训练的高效替代方案。还提出了下一步的建议，包括评估、偏好对齐、量化和部署。

希望这篇指南对你有所帮助。如果你对大语言模型感兴趣，建议查看 LLM Course。