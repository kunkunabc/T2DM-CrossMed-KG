import os
import sys
import importlib.util
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
import time
import types  # 用于更规范的方法绑定

# ================= 配置区 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULT_DIR = os.path.join(PROJECT_DIR, "results", "predictions")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "sensitivity_results")
PATH_SCRIPT_04 = os.path.join(SCRIPT_DIR, "04_prepare_anyburl.py")

# 实验参数
PENALTY_VALUES = [0.1, 0.2, 0.3, 0.6, 0.8, 1.0]  # 自变量
SAMPLE_SIZE = 20  # 测试药物数量
TOP_K = 10  # 评估 Top-K

# 目标药物 (混合中药和西药)
TARGET_DRUGS = [
    "HEB1435", "DCP11264",  # 黄蜀葵花, Topiramate
]


# ================= 工具函数 =================

def import_path_explainer():
    """动态导入 04 脚本中的 PathExplainer 类"""
    if not os.path.exists(PATH_SCRIPT_04):
        raise FileNotFoundError(f"未找到脚本: {PATH_SCRIPT_04}")
    spec = importlib.util.spec_from_file_location("mod04", PATH_SCRIPT_04)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PathExplainer


def load_drugs_from_file(file_path):
    """辅助函数：尝试读取文件并提取 entity 列"""
    drugs = []
    if not os.path.exists(file_path):
        print(f"[警告] 文件不存在: {file_path}")
        return drugs

    try:
        # 使用 engine='python' 和 sep=None 自动检测分隔符(逗号或Tab)
        df = pd.read_csv(file_path, sep=None, engine='python')

        # 清洗列名（防止列名带有空格）
        df.columns = [c.strip() for c in df.columns]

        if 'entity' in df.columns:
            # 提取药物ID，转为字符串并去重
            found = df['entity'].dropna().astype(str).unique().tolist()
            print(f"[信息] 从 {os.path.basename(file_path)} 加载了 {len(found)} 个药物")
            drugs.extend(found)
        else:
            print(f"[错误] 文件 {os.path.basename(file_path)} 中未找到 'entity' 列。现有列: {list(df.columns)}")

    except Exception as e:
        print(f"[错误] 读取文件 {os.path.basename(file_path)} 失败: {e}")

    return drugs


# ================= 主流程 =================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] 启动 DFS 惩罚因子校准实验...")

    # 1. 初始化 Explainer
    PathExplainer = import_path_explainer()
    kg_file = os.path.join(DATA_DIR, "final_unique_merged_triples.txt")
    rule_dir = os.path.join(PROJECT_DIR, "anyburl", "rules")

    print("正在初始化 PathExplainer...")
    try:
        explainer = PathExplainer(kg_file, rule_dir, disease_prefixes=["DM06"])
    except TypeError:
        # 兼容旧版本参数
        explainer = PathExplainer(kg_file, rule_dir)

    # 2. 准备测试药物样本
    all_drugs = set(TARGET_DRUGS)
    tcm_file = os.path.join(RESULT_DIR, "Top100_tcm.tsv")
    compound_file = os.path.join(RESULT_DIR, "Top100_western.tsv")

    # 从文件加载真实药物ID以补足样本
    loaded_drugs_pool = []

    # 分别读取两个文件
    loaded_drugs_pool.extend(load_drugs_from_file(tcm_file))
    loaded_drugs_pool.extend(load_drugs_from_file(compound_file))

    # 去重
    loaded_drugs_pool = list(set(loaded_drugs_pool))

    # 随机补足样本
    current_count = len(all_drugs)
    if current_count < SAMPLE_SIZE and loaded_drugs_pool:
        needed = SAMPLE_SIZE - current_count
        # 排除已经是 Target 的药物
        candidates = [d for d in loaded_drugs_pool if d not in all_drugs]

        if len(candidates) > 0:
            actual_needed = min(needed, len(candidates))
            sampled = np.random.choice(candidates, actual_needed, replace=False)
            all_drugs.update(sampled)
            print(f"[信息] 随机补充了 {len(sampled)} 个药物到测试集")
        else:
            print("[警告] 候选池中没有额外的新药物可供补充")

    test_drugs = list(all_drugs)
    print(f"最终测试药物样本数: {len(test_drugs)}")
    print(f"样本列表: {test_drugs[:5]}... (展示前5个)")

    # 3. 开始循环测试不同的 Penalty
    results = []

    # 定义 Patch 闭包工厂
    def make_patched_score_path(current_penalty):
        # 这里的 self 将在绑定时传入
        def patched_score_path(self, path, rule_confidence=1.0, is_rule_based=False):
            if len(path) < 3 or (len(path) % 2) != 1:
                return 0.0, {"error": "Invalid"}

            num_relations = (len(path) - 1) // 2
            length_decay_factor = self.get_length_decay(num_relations)

            product = 1.0
            for i in range(0, len(path) - 1, 2):
                rel = path[i + 1]
                entity = path[i + 2]

                # 权重获取逻辑
                rel_weight = self.relation_weights.get(rel, self.relation_weights.get("__DEFAULT__", 0.5))
                ent_type = self.get_entity_type(entity)
                base_ent_w = self.entity_type_weights.get(ent_type, 0.7)

                final_ent_w = base_ent_w
                if entity.startswith("INT"):
                    isi = self.ingredient_specificity.get(entity, 1.0)
                    final_ent_w *= isi

                product *= (rel_weight * final_ent_w)

            geo_mean = product ** (1.0 / num_relations) if num_relations > 0 else 0.0

            # === 关键修改点: 使用闭包中的 current_penalty ===
            if is_rule_based:
                eff_conf = min(1.0, float(rule_confidence))
            else:
                base_score = geo_mean * length_decay_factor
                eff_conf = min(1.0, base_score * current_penalty)
            # ============================================

            bonus = 1.0
            path_nodes = set(path[::2])
            if any(n.startswith("GEE") or n.startswith("INT") for n in path_nodes):
                bonus *= 1.2

            exp_rels = {"experiments", "includes", "experiments_transferred"}
            path_rels = set(path[1::2])
            if not path_rels.isdisjoint(exp_rels):
                bonus *= 1.8

            raw_score = geo_mean * length_decay_factor * eff_conf * bonus
            final_score = max(0.0, min(raw_score, 1.0))

            return final_score, {}

        return patched_score_path

    for penalty in PENALTY_VALUES:
        print(f"\n>>> Testing DFS Penalty: {penalty}")

        # --- 动态修改 Explainer 的 score_path ---
        # 使用 types.MethodType 将函数绑定为实例方法
        explainer.score_path = types.MethodType(make_patched_score_path(penalty), explainer)

        # --- 运行测试 ---
        total_rule_count = 0
        total_paths = 0
        total_score_sum = 0.0

        for drug in tqdm(test_drugs, desc=f"Penalty {penalty}"):
            # 强制清空缓存
            if hasattr(explainer, 'rule_path_cache'):
                explainer.rule_path_cache = {}

            # 获取 Top-K 路径
            try:
                paths = explainer.get_paths_for_drug(drug, need_top_k=TOP_K)
            except Exception as e:
                # 防止单个药物报错中断整个循环
                # print(f"Error processing {drug}: {e}")
                continue

            for p in paths:
                # 兼容返回格式: 可能是 (path, source, score) 或其他
                # 假设 p = (path_list, source_tag, score, details)
                if len(p) >= 3:
                    source_tag = p[1]
                    score = p[2]

                    if source_tag != "BFS_PATH":
                        total_rule_count += 1

                    total_paths += 1
                    total_score_sum += score

        # --- 统计 ---
        avg_rule_ratio = (total_rule_count / total_paths) if total_paths > 0 else 0.0
        avg_score = (total_score_sum / total_paths) if total_paths > 0 else 0.0

        print(f"  -> Rule Ratio: {avg_rule_ratio:.2%}")
        print(f"  -> Avg Score: {avg_score:.4f}")

        results.append({
            "DFS_Penalty": penalty,
            "Rule_Coverage_Ratio": avg_rule_ratio,
            "Average_Score": avg_score
        })

    # 4. 保存结果
    df_res = pd.DataFrame(results)
    out_file = os.path.join(OUTPUT_DIR, "dfs_penalty_calibration_results.csv")
    df_res.to_csv(out_file, index=False)
    print(f"\n校准实验完成！结果已保存至: {out_file}")


if __name__ == "__main__":
    main()