#05_sensitivity_experiment.py
import os
import importlib.util
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time
import re

# ================= 配置区 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULT_DIR = os.path.join(PROJECT_DIR, "results", "predictions")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "sensitivity_results")

# 【重要】请确保文件名与你实际的04脚本文件名一致
PATH_SCRIPT_04 = os.path.join(SCRIPT_DIR, "04_prepare_anyburl.py")

# 目标药物列表 (包含中药和西药)
TARGET_DRUGS = [
    "HEB1435", "HEB1482", "HEB5161", "HEB3348", "HEB0006", "HEB5184", "HEB5058",
    "HEB4619", "HEB2834", "HEB2050", "DCP11264", "DCP01226", "DCP04678", "DCP12738",
    "DCP10156", "DCP07553", "DCP10157", "DCP11226", "DCP12927", "DCP11343"
]

SAMPLE_SIZE = 20
EVAL_TOP_K = 10
POOL_SIZE = 500  # 候选池大小，越大越能体现重排序效果

# ================= 实验参数配置 =================

BASELINE_CONFIG = {
    "name": "Baseline",
    "global_weight_factor": 1.0,  # 正常使用字典权重
    "dfs_penalty": 0.2,  # DFS 惩罚 (与04/07一致)
    "bonus_exp": 1.8,
    "isi_floor": 0.01,
    "decay_factor": 0.9,  # 基础衰减因子
    "mode": "normal"
}

EXP_CONFIGS = [
    BASELINE_CONFIG,

    # Exp1: 权重区分度 (Weight Distinction)
    {
        "name": "Exp1_NoDistinction",
        "global_weight_factor": 1.0,
        "dfs_penalty": 1.0,  # 取消惩罚，所有权重归一
        "bonus_exp": 1.8, "isi_floor": 0.01, "decay_factor": 0.9,
        "mode": "no_distinction"
    },
    {
        "name": "Exp1_StrongDistinction",
        "global_weight_factor": 0.2,  # 拉大权重差异
        "dfs_penalty": 0.05,
        "bonus_exp": 1.8, "isi_floor": 0.01, "decay_factor": 0.9,
        "mode": "strong_distinction"
    },

    # Exp2: 长度衰减 (Length Decay)
    {
        "name": "Exp2_NoDecay",
        "global_weight_factor": 1.0, "dfs_penalty": 0.2, "bonus_exp": 1.8, "isi_floor": 0.01,
        "decay_factor": 1.0,  # 无衰减
        "mode": "normal"
    },
    {
        "name": "Exp2_StrongDecay",
        "global_weight_factor": 1.0, "dfs_penalty": 0.2, "bonus_exp": 1.8, "isi_floor": 0.01,
        "decay_factor": 0.5,  # 强衰减
        "mode": "normal"
    },

    # Exp3: 奖励机制 (Bonus)
    {
        "name": "Exp3_NoBonus",
        "global_weight_factor": 1.0, "dfs_penalty": 0.2,
        "bonus_exp": 1.0,  # 无奖励
        "isi_floor": 0.01, "decay_factor": 0.9, "mode": "normal"
    },
    {
        "name": "Exp3_HighBonus",
        "global_weight_factor": 1.0, "dfs_penalty": 0.2,
        "bonus_exp": 5.0,  # 超高奖励
        "isi_floor": 0.01, "decay_factor": 0.9, "mode": "normal"
    },

    # Exp4: 成分特异性 (ISI)
    {
        "name": "Exp4_NoISI",
        "global_weight_factor": 1.0, "dfs_penalty": 0.2, "bonus_exp": 1.8,
        "isi_floor": 1.0,  # ISI 失效 (全部为1.0)
        "decay_factor": 0.9, "mode": "normal"
    }
]

# 完整权重字典 (与 04/07 保持完全一致)
RELATION_WEIGHTS_DICT = {
    "treats": 0.7, "causes": 0.7, "palliates": 0.7, "includes": 0.9,
    "binds": 0.9, "regulates": 0.9, "candidate_for": 0.9,
    "downregulates": 0.9, "upregulates": 0.9, "participates": 0.9,
    "associates": 0.9, "expresses": 0.9, "fusion": 0.9,
    "coexpression": 0.9, "experiments": 1.0, "database": 0.9,
    "textmining": 0.9, "cooccurrence": 0.9, "homology": 0.9,
    "interacts": 0.9, "functions": 0.9, "is_associated_with": 0.9,
    "acts_on": 0.9, "coexpression_transferred": 0.9,
    "database_transferred": 0.9, "experiments_transferred": 1.0,
    "__DEFAULT__": 0.7
}

ENTITY_WEIGHTS_DICT = {
    "DiseaseMM": 0.7, "DiseaseTCM": 0.7, "Gene": 1.0, "Compound": 1.0,
    "Herb": 1.0, "Formula": 0.7, "SymptomMM": 0.7, "SymptomTCM": 0.7,
    "Pathomechanism": 0.7, "Anatomy": 0.7, "Ingredient": 1.0, "GO": 0.7,
    "PharmClass": 0.7, "Pathway": 0.7, "Book": 0.7, "TextID": 0.7, "Chapter": 0.7,
    "__DEFAULT__": 0.7
}


# ================= 核心工具函数 =================

def import_path_explainer():
    """动态导入 04 脚本中的 PathExplainer 类"""
    if not os.path.exists(PATH_SCRIPT_04):
        raise FileNotFoundError(f"未找到脚本: {PATH_SCRIPT_04}")
    spec = importlib.util.spec_from_file_location("mod04", PATH_SCRIPT_04)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PathExplainer


def format_path_full(raw_path):
    if not raw_path: return ""
    elements = []
    for i, item in enumerate(raw_path):
        if i % 2 == 1:
            elements.append(f"-[{item}]->")
        else:
            elements.append(str(item))
    return " ".join(elements)


def extract_confidence(rule_info):
    if not rule_info or rule_info == "DFS": return None
    try:
        # 尝试从 "Rule:ID (0.85)" 这种格式中提取置信度
        # 或者从 tab 分隔的字符串中提取
        parts = re.split(r'[\t\s]+', str(rule_info).strip())
        for p in parts:
            if re.match(r'^0\.\d+$', p): return float(p)
    except:
        pass
    return None


def calculate_score_dynamic(path, explainer, config, rule_confidence=None):
    """
    动态评分函数：根据 config 实时计算路径分数，用于重排序。
    """
    mode = config['mode']
    dfs_penalty_val = config['dfs_penalty']
    global_weight_factor = config['global_weight_factor']
    decay_val = config['decay_factor']
    bonus_exp_val = config['bonus_exp']
    isi_floor_val = config['isi_floor']

    num_relations = (len(path) - 1) // 2

    # 1. Length Decay (严格对齐 04/07 的分段衰减逻辑)
    if num_relations <= 2:
        length_decay = 1.0
    elif num_relations <= 4:
        length_decay = decay_val ** (num_relations - 2)
    else:
        # 如果是 Baseline 或 decay 为默认值，使用 0.7 的强衰减 (长路径抑制)
        if config['name'] == 'Baseline' or config['decay_factor'] == 0.9:
            length_decay = 0.7 ** (num_relations - 2)
        else:
            # 实验组：根据 decay_val 动态调整
            length_decay = (decay_val * 0.8) ** (num_relations - 2)

    # 2. Geometric Mean (权重计算)
    product = 1.0

    for i in range(0, len(path) - 1, 2):
        rel = path[i + 1]
        entity = path[i + 2]

        # 关系权重
        if mode == "no_distinction":
            r_w = 1.0
        elif mode == "strong_distinction":
            orig_w = RELATION_WEIGHTS_DICT.get(rel, 0.7)
            r_w = orig_w * global_weight_factor
        else:
            r_w = RELATION_WEIGHTS_DICT.get(rel, 0.7)

        # 实体权重
        ent_type = explainer.get_entity_type(entity)
        if mode == "no_distinction":
            base_e_w = 1.0
        else:
            base_e_w = ENTITY_WEIGHTS_DICT.get(ent_type, 0.7)
            if mode == "strong_distinction":
                base_e_w *= global_weight_factor

        final_e_w = base_e_w

        # ISI 特异性权重
        if entity.startswith("INT"):
            # 如果 isi_floor_val 是 1.0 (Exp4_NoISI)，则强制为 1.0
            if isi_floor_val >= 0.99:
                isi_factor = 1.0
            else:
                # 获取真实 ISI
                raw_isi = explainer.ingredient_specificity.get(entity, 1.0)
                isi_factor = raw_isi
            final_e_w *= isi_factor

        product *= (r_w * final_e_w)

    geometric_mean = product ** (1.0 / num_relations) if num_relations > 0 else 0.0

    # 3. Confidence Logic (规则置信度 vs DFS 惩罚)
    if rule_confidence is not None:
        effective_confidence = min(1.0, float(rule_confidence))
    else:
        base_confidence = geometric_mean * length_decay
        effective_confidence = base_confidence * dfs_penalty_val

    # 4. Bonus (奖励机制)
    bonus = 1.0
    path_nodes = set(path[::2])
    path_rels = set(path[1::2])

    # 机制奖励 (Gene/Ingredient)
    if any(n.startswith("GEE") or n.startswith("INT") for n in path_nodes):
        bonus *= 1.2

    # 证据奖励 (Experiments)
    exp_rels_set = {"experiments", "includes", "experiments_transferred"}
    if not path_rels.isdisjoint(exp_rels_set):
        bonus *= bonus_exp_val

    final_score = geometric_mean * length_decay * effective_confidence * bonus
    return max(0.0, min(final_score, 1.0))


def calculate_iou(baseline_ranks, variant_ranks, top_k):
    """计算两个排名列表 Top-K 的交并比 (IoU)"""
    baseline_top = set(p[0] for p in baseline_ranks[:top_k])
    variant_top = set(p[0] for p in variant_ranks[:top_k])

    # 注意：p[0] 是 path_str，唯一标识路径
    inter = len(baseline_top & variant_top)
    union = len(baseline_top | variant_top)
    return inter / union if union > 0 else 0.0


# ================= 主流程 =================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("正在导入 PathExplainer (从 #04 脚本)...")
    PathExplainer = import_path_explainer()

    kg_file = os.path.join(DATA_DIR, "final_unique_merged_triples.txt")
    rule_dir = os.path.join(PROJECT_DIR, "anyburl", "rules")

    # 初始化解释器 (会自动计算 ISI)
    try:
        explainer = PathExplainer(kg_file, rule_dir, disease_prefixes=["DM06"])
    except TypeError:  # 兼容旧版本构造函数
        explainer = PathExplainer(kg_file, rule_dir)

    all_target_drugs = set(TARGET_DRUGS)
    # 尝试补充随机药物以达到 SAMPLE_SIZE (如果有 tcm_file)
    tcm_file = os.path.join(RESULT_DIR, "top100_tcm.tsv")
    if os.path.exists(tcm_file) and len(all_target_drugs) < SAMPLE_SIZE:
        try:
            df = pd.read_csv(tcm_file, sep='\t')
            # 排除已有的
            candidates = df[~df['entity'].isin(all_target_drugs)]['entity'].unique()
            need = SAMPLE_SIZE - len(all_target_drugs)
            if len(candidates) > need:
                random_drugs = np.random.choice(candidates, need, replace=False)
                all_target_drugs.update(random_drugs)
        except:
            pass

    test_drugs = list(all_target_drugs)
    print(f"[{time.strftime('%H:%M:%S')}] 开始敏感性分析，共 {len(test_drugs)} 个药物...")

    all_metrics = []
    all_case_studies = []

    for drug in tqdm(test_drugs, desc="Processing Drugs"):
        try:
            # 获取路径池 (这里调用的是 04 脚本的方法，会用到 Beam=100 和 VIP 策略)
            # 返回格式: [(path_list, source_info, score, details), ...]
            raw_paths_data = explainer.get_paths_for_drug(drug, need_top_k=POOL_SIZE)
        except Exception as e:
            # print(f"Error processing {drug}: {e}")
            continue

        if not raw_paths_data: continue

        # 构建统一的路径对象池
        path_pool = []
        seen_paths = set()
        for item in raw_paths_data:
            path_tuple = tuple(item[0])
            if path_tuple not in seen_paths:
                conf = extract_confidence(item[1])
                path_pool.append({
                    "path": item[0],
                    "rule_info": item[1],
                    "confidence": conf
                })
                seen_paths.add(path_tuple)

        # --------------------------------------------------
        # 1. 计算 Baseline Ranks (基准)
        # --------------------------------------------------
        baseline_ranks = []
        for item in path_pool:
            score = calculate_score_dynamic(item['path'], explainer, BASELINE_CONFIG,
                                            rule_confidence=item['confidence'])
            path_str = format_path_full(item['path'])
            baseline_ranks.append((path_str, score, item))

        # 排序：先按分数降序，再按路径字符串降序 (保证确定性)
        baseline_ranks.sort(key=lambda x: (x[1], x[0]), reverse=True)

        # --------------------------------------------------
        # 2. 遍历所有实验变体
        # --------------------------------------------------
        for config in EXP_CONFIGS:
            cfg_name = config['name']
            current_ranks = []
            for item in path_pool:
                score = calculate_score_dynamic(item['path'], explainer, config, rule_confidence=item['confidence'])
                path_str = format_path_full(item['path'])
                current_ranks.append((path_str, score, item))

            # 排序
            current_ranks.sort(key=lambda x: (x[1], x[0]), reverse=True)

            # 计算 IoU (Top-K)
            iou = calculate_iou(baseline_ranks, current_ranks, top_k=EVAL_TOP_K)

            # 记录指标
            all_metrics.append({
                "Experiment": cfg_name, "Drug": drug, "IoU": iou,
                **{k: v for k, v in config.items() if k != 'name'}
            })

            # 记录 Case Study (仅针对特定药物，避免文件过大)
            if drug in TARGET_DRUGS:
                for rank, (p_str, s, origin_item) in enumerate(current_ranks[:10], 1):
                    # 标记来源：Rule 还是 DFS
                    source_raw = str(origin_item['rule_info'])
                    source_type = "Rule" if ("Rule" in source_raw or "BFS" in source_raw) else "DFS"

                    all_case_studies.append({
                        "Drug": drug, "Experiment": cfg_name, "Rank": rank,
                        "Score": f"{s:.4f}", "Source": source_type, "Path": p_str
                    })

    # ================= 保存结果 =================

    # 1. 保存 Metrics
    df_metrics = pd.DataFrame(all_metrics)
    if not df_metrics.empty:
        # [Raw Data] 保存每一行数据，用于画箱线图 (#06脚本需要这个文件)
        raw_path = os.path.join(OUTPUT_DIR, "sensitivity_metrics_raw.csv")
        df_metrics.to_csv(raw_path, index=False)
        print(f"原始数据已保存: {raw_path}")

        # [Summary Data] 保存汇总平均值
        order_map = {cfg['name']: i for i, cfg in enumerate(EXP_CONFIGS)}
        df_summary = df_metrics.groupby("Experiment").agg({
            "IoU": "mean"
        }).reset_index()
        df_summary['order'] = df_summary['Experiment'].map(order_map)
        df_summary = df_summary.sort_values('order').drop(columns=['order'])

        sum_path = os.path.join(OUTPUT_DIR, "sensitivity_metrics_final.csv")
        df_summary.to_csv(sum_path, index=False)
        print(f"汇总数据已保存: {sum_path}")

    # 2. 保存 Case Studies
    df_cases = pd.DataFrame(all_case_studies)
    if not df_cases.empty:
        case_path = os.path.join(OUTPUT_DIR, "case_study_paths_final.csv")
        df_cases.to_csv(case_path, index=False)
        print(f"案例详情已保存: {case_path}")

    print(f"全部完成! 敏感性分析结束。")


if __name__ == "__main__":
    main()