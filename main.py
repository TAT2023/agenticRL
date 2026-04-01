import json
from hello_agents.tools import RLTrainingTool

# 创建 RL 训练工具
rl_tool = RLTrainingTool()

# ============================
# 1. SFT 训练（本地模型 + 本地数据）
# ============================
sft_result_str = rl_tool.run({
    "action": "train",
    "algorithm": "sft",
    "model_name": "./models/Qwen3-0.6B",      # ⭐ 本地模型
    "output_dir": "./models/quick_test_sft",
    "max_samples": 10,
    "num_epochs": 1,
    "batch_size": 2,
    "use_lora": True,
    "local_data_path": "./data/gsm8k/train.jsonl"   # ⭐ 本地数据
})

sft_result = json.loads(sft_result_str)
print(f"\n✓ SFT 训练完成，模型保存在: {sft_result.get('output_dir')}")

# ============================
# 2. GRPO 训练（本地模型 + 本地数据）
# ============================
grpo_result_str = rl_tool.run({
    "action": "train",
    "algorithm": "grpo",
    "model_name": "./models/Qwen3-0.6B",      # ⭐ 本地模型
    "output_dir": "./models/quick_test_grpo",
    "max_samples": 5,
    "num_epochs": 1,
    "batch_size": 2,
    "use_lora": True,
    "local_data_path": "./data/gsm8k/train.jsonl"   # ⭐ 本地数据
})

grpo_result = json.loads(grpo_result_str)
print(f"\n✓ GRPO 训练完成，模型保存在: {grpo_result.get('output_dir')}")

# ============================
# 3. 评估模型（本地测试集）
# ============================
eval_result_str = rl_tool.run({
    "action": "evaluate",
    "model_path": "./models/quick_test_grpo",
    "max_samples": 10,
    "use_lora": True,
    "local_data_path": "./data/gsm8k/test.jsonl"    # ⭐ 本地测试集
})

eval_result = json.loads(eval_result_str)
print("\n✓ 评估完成:")
print(f"  - 准确率: {eval_result.get('accuracy')}")
print(f"  - 平均奖励: {eval_result.get('average_reward')}")
print(f"  - 测试样本数: {eval_result.get('num_samples')}")

print("\n" + "=" * 50)
print("🎉 恭喜！你已经完成了第一个 Agentic RL 模型的训练！")
print("=" * 50)
