import os
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import importlib.util

# ==============================================================================
#                                USER CONFIG
# ==============================================================================
CONFIG = {
    "TARGET_DISEASES": ["DM06"],
    # 目标药物列表
    "TARGET_DRUGS": [
        "HEB1435", "HEB1482", "HEB5161", "HEB3348", "HEB0006", "HEB5184", "HEB5058",
        "HEB4619", "HEB2834", "HEB2050", "DCP11264", "DCP01226", "DCP04678", "DCP12738",
        "DCP10156", "DCP07553", "DCP10157", "DCP11226", "DCP12927", "DCP11343"
    ],
    "TOP_K_PATHS": 10,
    "MAX_HOP": 4
}

# ===================== 动态导入 #04 脚本 =====================
# 确保 04 脚本就在同级目录下
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_SCRIPT_04 = os.path.join(SCRIPT_DIR, "04_prepare_anyburl.py")


def import_path_explainer_class():
    """从 04 脚本中动态导入 PathExplainer 类，确保逻辑 100% 同步"""
    if not os.path.exists(PATH_SCRIPT_04):
        raise FileNotFoundError(f"未找到核心脚本: {PATH_SCRIPT_04}，请确保它与本脚本在同一目录下。")
    spec = importlib.util.spec_from_file_location("mod04", PATH_SCRIPT_04)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PathExplainer


# 导入基类
BasePathExplainer = import_path_explainer_class()

# ===================== 日志设置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ===================== 继承并扩展 PathExplainer =====================
class AblationPathExplainer(BasePathExplainer):
    """
    继承自 04 脚本的 PathExplainer。
    只添加消融实验特有的 'set_ablation_mode' 逻辑，
    搜索、评分、规则匹配等核心逻辑完全复用父类 (即复用 Beam=100+VIP 策略)。
    """

    def __init__(self, kg_file, rule_dir):
        # 调用父类初始化
        # 注意：这里会复用父类的权重字典、ISI 计算等
        super().__init__(kg_file, rule_dir, disease_prefixes=CONFIG["TARGET_DISEASES"])

        # 【关键】备份原始参数，以便消融模式切换时恢复
        # 注意：父类必须已经初始化了这些属性
        self.raw_relation_weights = self.relation_weights.copy()
        self.raw_entity_type_weights = self.entity_type_weights.copy()
        self.raw_ingredient_specificity = self.ingredient_specificity.copy()

        # 默认 DFS 惩罚 (硬编码对齐 04 脚本中的 0.2)
        self.default_dfs_penalty = 0.2
        # 我们需要动态修改 score_path 中的行为，但父类的 score_path 里 dfs_penalty 是硬编码的 0.2
        # 为了解决这个问题，我们需要稍微"魔改"一下父类的行为，或者在调用 score_path 前动态修改类属性
        # 但最稳妥的方式是：利用我们覆盖的 score_path 方法（见下文）

        self.current_dfs_penalty = self.default_dfs_penalty
        self.use_rules_flag = True
        self.use_bonus_flag = True

    def set_ablation_mode(self, mode):
        """切换消融模式，修改内部权重状态"""
        # 1. 重置为 Full Model 状态
        self.relation_weights = self.raw_relation_weights.copy()
        self.entity_type_weights = self.raw_entity_type_weights.copy()
        self.ingredient_specificity = self.raw_ingredient_specificity.copy()
        self.current_dfs_penalty = self.default_dfs_penalty
        self.use_rules_flag = True
        self.use_bonus_flag = True

        # 2. 应用消融设置
        if mode == "w/o ISI":
            # 将ISI全部设为1.0
            self.ingredient_specificity = {k: 1.0 for k in self.raw_ingredient_specificity}
            logger.info(f"  [Mode] {mode}: ISI -> 1.0")

        elif mode == "w/o Weights":
            # 权重归一
            self.relation_weights = {k: 1.0 for k in self.relation_weights}
            self.entity_type_weights = {k: 1.0 for k in self.entity_type_weights}
            self.current_dfs_penalty = 1.0  # 无权重时，取消 DFS 惩罚
            logger.info(f"  [Mode] {mode}: Weights/Penalty -> 1.0")

        elif mode == "w/o Rules":
            self.use_rules_flag = False
            self.current_dfs_penalty = 1.0  # 无规则时，DFS 是唯一来源，不惩罚
            logger.info(f"  [Mode] {mode}: Rules -> Disabled")

        elif mode == "w/o Bonus":
            self.use_bonus_flag = False
            logger.info(f"  [Mode] {mode}: Bonus -> Disabled")

        elif mode == "Full Model":
            logger.info(f"  [Mode] {mode}: Standard Setup")

    # 【重要】重写 score_path 以支持动态 DFS 惩罚和 Bonus 开关
    # 虽然父类有这个方法，但父类的 dfs_penalty=0.2 是写死的，且没有 use_bonus_flag 开关
    # 所以必须在这里覆盖它，保持计算逻辑一致，但参数可变。
    def score_path(self, path, rule_confidence=1.0, is_rule_based=False):
        if len(path) < 3 or (len(path) % 2) != 1: return 0.0, {}

        num_rels = (len(path) - 1) // 2
        # 长度衰减 (保持与04一致)
        if num_rels <= 2:
            length_decay = 1.0
        elif num_rels <= 4:
            length_decay = 0.9 ** (num_rels - 2)
        else:
            length_decay = 0.7 ** (num_rels - 2)

        product = 1.0
        for i in range(0, len(path) - 1, 2):
            rel = path[i + 1]
            ent = path[i + 2]

            w_r = self.relation_weights.get(rel, 0.7)
            ent_type = self.get_entity_type(ent)
            w_e = self.entity_type_weights.get(ent_type, 0.7)

            if ent.startswith("INT"):
                w_e *= self.ingredient_specificity.get(ent, 1.0)

            product *= w_r * w_e

        geo_mean = product ** (1.0 / num_rels)

        if is_rule_based:
            eff_conf = min(1.0, float(rule_confidence))
        else:
            # 使用动态的 penalty
            eff_conf = min(1.0, geo_mean * length_decay * self.current_dfs_penalty)

        bonus = 1.0
        if self.use_bonus_flag:
            path_nodes = set(path[::2])
            if any(e.startswith("GEE") or e.startswith("INT") for e in path_nodes): bonus *= 1.2
            path_rels = set(path[1::2])
            strong_ev = {"experiments", "includes", "experiments_transferred"}
            if any(r in strong_ev for r in path_rels): bonus *= 1.8

        final_score = max(0.0, min(geo_mean * length_decay * eff_conf * bonus, 1.0))
        # 构造简单返回，不需要 detail
        return final_score, {'score': final_score}

    # 重写 find_rule_based_path 以支持开关
    def find_rule_based_path(self, drug):
        if not self.use_rules_flag: return []
        return super().find_rule_based_path(drug)


# ===================== 实验运行逻辑 =====================
def run_ablation_study():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    kg_file = os.path.join(project_dir, "data", "final_unique_merged_triples.txt")
    rule_dir = os.path.join(project_dir, "anyburl", "rules")
    output_dir = os.path.join(project_dir, "results", "ablation_study")
    os.makedirs(output_dir, exist_ok=True)

    print("正在初始化 Explainer (基于 #04 脚本)...")
    try:
        # 使用继承的类
        explainer = AblationPathExplainer(kg_file, rule_dir)
    except Exception as e:
        logger.error(f"Init error: {e}");
        return

    target_drugs = CONFIG["TARGET_DRUGS"]
    top_k = CONFIG["TOP_K_PATHS"]
    variants = ["Full Model", "w/o ISI", "w/o Weights", "w/o Rules", "w/o Bonus"]
    metrics_data = defaultdict(dict)

    print("\n" + "=" * 80 + "\nSTARTING ABLATION STUDY EXPERIMENT\n" + "=" * 80)

    for variant in variants:
        print(f"\n>>> Running Variant: [{variant}]")
        explainer.set_ablation_mode(variant)

        file_name = f"paths_{variant.replace(' ', '_').replace('/', '')}.tsv"
        file_path = os.path.join(output_dir, file_name)

        var_isi, var_ev, var_rule, var_mech = [], [], [], []

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["Drug", "Rank", "Score", "Type", "Path"])

            pbar = tqdm(target_drugs, desc=f"Variant {variant}", unit="drug")
            for drug in pbar:
                # 调用父类的 get_paths_for_drug (享受 Beam=100 和 VIP)
                # 父类会回调子类的 score_path 和 find_rule_based_path
                top_paths = explainer.get_paths_for_drug(drug, need_top_k=top_k)

                if not top_paths: continue

                for rank, p in enumerate(top_paths):
                    path_str = " -> ".join(p[0])
                    writer.writerow([drug, rank + 1, f"{p[2]:.4f}", p[1], path_str])

                # 指标计算 (使用原始真实值)
                isi_vals = []
                ev_rels, tot_rels = 0, 0
                rule_hit, mech_hit_count = 0, 0
                strong_ev = {"experiments", "includes", "experiments_transferred"}
                num_paths = len(top_paths)

                for p in top_paths:
                    path_list = p[0]
                    # ISI
                    for ent in path_list[::2]:
                        if ent.startswith("INT"):
                            # 始终查原始ISI表
                            val = explainer.raw_ingredient_specificity.get(ent, 0.2)
                            isi_vals.append(val)
                    # Evid
                    for rel in path_list[1::2]:
                        tot_rels += 1
                        if rel in strong_ev: ev_rels += 1
                    # Rule
                    if "Rule" in str(p[1]) or "BFS" not in str(p[1]): rule_hit += 1
                    # Mech
                    mech_nodes = sum(1 for e in path_list[::2] if e.startswith("GEE") or e.startswith("INT"))
                    if len(path_list[::2]) > 0:
                        mech_hit_count += (mech_nodes / len(path_list[::2]))

                drug_avg_isi = sum(isi_vals) / len(isi_vals) if isi_vals else 0
                var_isi.append(drug_avg_isi)
                var_ev.append(ev_rels / tot_rels if tot_rels > 0 else 0)
                var_rule.append(rule_hit / num_paths if num_paths > 0 else 0)
                var_mech.append(mech_hit_count / num_paths if num_paths > 0 else 0)

        def mean(l):
            return sum(l) / len(l) if l else 0.0

        metrics_data[variant]["Avg_ISI"] = mean(var_isi)
        metrics_data[variant]["Evidence_Ratio"] = mean(var_ev)
        metrics_data[variant]["Rule_Ratio"] = mean(var_rule)
        metrics_data[variant]["Mech_Density"] = mean(var_mech)

    print("\n" + "-" * 90 + "\nABLATION RESULTS SUMMARY TABLE\n" + "-" * 90)
    print(f"{'Variant':<15} | {'Avg. ISI':<10} | {'Evid. Ratio':<12} | {'Rule %':<10} | {'Mech. %':<10}")
    print("-" * 90)
    for var in variants:
        d = metrics_data[var]
        print(
            f"{var:<15} | {d['Avg_ISI']:.4f}     | {d['Evidence_Ratio']:.2%}       | {d['Rule_Ratio']:.2%}     | {d['Mech_Density']:.2%}")

    plot_ablation_chart(metrics_data, variants, output_dir)


def plot_ablation_chart(data, variants, out_dir):
    try:
        keys = ["Avg_ISI", "Evidence_Ratio", "Rule_Ratio", "Mech_Density"]
        titles = ["Avg Path Specificity", "Evidence Ratio", "Rule Contribution", "Mechanism Density"]
        plt.rcParams['font.family'] = 'Arial'
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for i, key in enumerate(keys):
            ax = axes[i // 2, i % 2]
            vals = [data[v].get(key, 0) for v in variants]
            x = np.arange(len(variants))
            colors = ['#2c3e50' if v == 'Full Model' else '#95a5a6' for v in variants]
            bars = ax.bar(x, vals, color=colors, alpha=0.8, edgecolor='black', width=0.6)
            ax.set_title(titles[i], fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(variants, rotation=20, ha='right')
            top_lim = max(vals) * 1.2 if vals and max(vals) > 0 else 1.0
            ax.set_ylim(0, top_lim)
            for bar in bars:
                h = bar.get_height()
                if h > 0: ax.text(bar.get_x() + bar.get_width() / 2, h + (top_lim * 0.02), f'{h:.3f}', ha='center',
                                  va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ablation_results.png"), dpi=300)
    except Exception:
        pass


if __name__ == "__main__":
    run_ablation_study()