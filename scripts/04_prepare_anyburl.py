#04_prepare_anyburl.py
import os
import re
import csv
import pickle
import math
import pandas as pd
import logging
import shutil
import gc
import time
import heapq
from collections import defaultdict, deque
from tqdm import tqdm
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 日志设置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("path_explainer_optimized.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PathExplainer:
    def __init__(self, kg_file, rule_dir, disease_prefixes=None):
        self.kg_file = kg_file
        self.rule_dir = rule_dir
        self.triples = self.load_triples(kg_file)
        self.rule_body_cache = {}
        logger.info(f"加载了 {len(self.triples)} 条三元组")

        self.index = self.build_index()
        logger.info("知识图谱索引构建完成")

        # ========= 新增：预计算中药成分特异性权重 (ISI) =========
        logger.info("正在计算中药成分特异性权重 (ISI)...")
        self.ingredient_specificity = self._calculate_ingredient_specificity()
        logger.info(f"成分特异性计算完成，共覆盖 {len(self.ingredient_specificity)} 个成分")

        rule_file = self.find_latest_rule_file()
        if not rule_file:
            # 允许在无规则文件的情况下运行（只跑DFS）
            logger.warning(f"Warning: No rule file found in {rule_dir}. Running in DFS-only mode.")
            self.rules = defaultdict(list)
        else:
            logger.info(f"解析规则文件: {rule_file}")
            self.rules = self.parse_rules(rule_file)
            if not self.rules:
                logger.error("未解析到任何规则!")
            else:
                top_rels = sorted(self.rules.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                logger.info(f"Top 5关系规则数: {', '.join([f'{k}({len(v)})' for k, v in top_rels])}")

        self.entity_prefix_map = self.create_entity_prefix_map()
        self.relation_whitelist = self.create_relation_whitelist()

        # ========= 目标疾病前缀 =========
        self.target_disease_prefixes = disease_prefixes or ["DM"]

        # ========= 权重配置 =========
        self.relation_weights = {
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
        self.entity_type_weights = {
            "DiseaseMM": 0.7, "DiseaseTCM": 0.7,
            "Gene": 1.0, "Compound": 1.0,
            "Herb": 1.0, "Formula": 0.7,
            "SymptomMM": 0.7, "SymptomTCM": 0.7,
            "Pathomechanism": 0.7, "Anatomy": 0.7,
            "Ingredient": 1.0, "GO": 0.7,
            "PharmClass": 0.7, "Pathway": 0.7,
            "Book": 0.7, "TextID": 0.7, "Chapter": 0.7,
            "__DEFAULT__": 0.7
        }

        # 疾病实体缓存
        self.disease_cache = set()
        self.populate_disease_cache()

        # 规则路径缓存
        self.rule_path_cache = {}
        self.rule_miss_cache = set()

        # KG二级索引
        self.build_kg_indexes()

        # 药物关系缓存
        self.drug_relations_cache = {}

        # 缓存清理和内存控制
        self.CACHE_CLEAN_INTERVAL = 15
        self.CACHE_RETENTION_DAYS = 7
        self.MEMORY_CLEAN_THRESHOLD = 90
        self.cache_lock = threading.Lock()

    # ================== 核心方法：计算成分特异性 (修正版) ==================
    def _calculate_ingredient_specificity(self):
        """
        基于 TF-IDF 思想计算中药成分(INT)的特异性权重。
        权重越低，说明该成分越通用（如槲皮素）；权重越高，说明越特异。
        """
        # 1. 统计每个成分出现在多少个中药(HEB)中
        herb_counts = set()
        ing_doc_freq = defaultdict(set)

        for h, r, t in self.triples:
            head_is_herb = h.startswith("HEB")
            tail_is_ing = t.startswith("INT")
            tail_is_herb = t.startswith("HEB")
            head_is_ing = h.startswith("INT")

            if head_is_herb: herb_counts.add(h)
            if tail_is_herb: herb_counts.add(t)

            if head_is_herb and tail_is_ing:
                ing_doc_freq[t].add(h)
            elif head_is_ing and tail_is_herb:
                ing_doc_freq[h].add(t)

        total_herbs = len(herb_counts)
        if total_herbs == 0:
            logger.warning("未检测到中药实体(HEB)，成分特异性权重将默认为1.0")
            return {}

        # 2. 计算 IDF 并归一化
        max_idf = math.log(total_herbs) if total_herbs > 1 else 1.0
        specificity_map = {}

        for ing, herbs_set in ing_doc_freq.items():
            df = len(herbs_set)
            idf = math.log(total_herbs / (df + 1))

            # 【核心修改点 1】ISI 下限调整为 0.01 (严苛模式，与敏感性分析对齐)
            normalized_weight = max(0.01, idf / max_idf)
            specificity_map[ing] = normalized_weight

        if specificity_map:
            sorted_specs = sorted(specificity_map.items(), key=lambda x: x[1])
            logger.info(f"Top 5 通用成分(低权重): {sorted_specs[:5]}")
            logger.info(f"Top 5 特异成分(高权重): {sorted_specs[-5:]}")

        return specificity_map

    # ================== 内存监控 ==================
    def memory_usage(self):
        try:
            return psutil.virtual_memory().percent
        except:
            return 0

    def check_memory_usage(self):
        mem_usage = self.memory_usage()
        if mem_usage > self.MEMORY_CLEAN_THRESHOLD:
            logger.warning(f"内存使用过高 ({mem_usage}%), 触发内存清理")
            self.rule_path_cache = {}
            gc.collect()
            logger.info("内存清理完成")

    # ================== I/O 和索引 ==================
    def load_triples(self, file_path):
        triples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) < 3: continue
                    h, r, t = parts[0], parts[1], parts[2]
                    triples.append((h, r, t))
            return triples
        except Exception as e:
            logger.error(f"加载三元组文件失败: {str(e)}")
            return []

    def build_index(self):
        index = {
            'head_rel': defaultdict(lambda: defaultdict(set)),
            'tail_rel': defaultdict(lambda: defaultdict(set)),
            'all_entities': set(),
            'all_triples': set(self.triples)
        }
        for h, r, t in self.triples:
            index['head_rel'][h][r].add(t)
            index['tail_rel'][t][r].add(h)
            index['all_entities'].add(h)
            index['all_entities'].add(t)
        return index

    def build_kg_indexes(self):
        self.kg_triples_set = set(self.triples)
        self.kg_index = defaultdict(lambda: defaultdict(set))
        for s, p, o in self.triples:
            self.kg_index[(s, p)]['objects'].add(o)
            self.kg_index[(s, None)]['relations'].add((p, o))

    def find_latest_rule_file(self):
        if not os.path.exists(self.rule_dir): return None
        rule_files = []
        for f in os.listdir(self.rule_dir):
            full_path = os.path.join(self.rule_dir, f)
            if os.path.isfile(full_path):
                match = re.search(r'(\d+)$', f)
                if match:
                    snapshot = int(match.group(1))
                    rule_files.append((snapshot, full_path))
        if not rule_files: return None
        rule_files.sort(key=lambda x: x[0], reverse=True)
        return rule_files[0][1]

    def create_entity_prefix_map(self):
        return {
            "DM06": "DiseaseMM", "DITCM02": "DiseaseTCM",
            "C": "SymptomMM", "D": "SymptomMM", "TS": "SymptomTCM",
            "BYBJ": "Pathomechanism", "FJ": "Formula", "HEB": "Herb",
            "GEE": "Gene", "DCP": "Compound", "GO": "GO",
            "INT": "Ingredient", "UBERON": "Anatomy", "SM": "Book",
            "TWID": "TextID", "N": "PharmClass", "ZJPM": "Chapter",
            "__DEFAULT__": "Pathway"
        }

    def create_relation_whitelist(self):
        return {
            "downregulates", "expresses", "upregulates", "resembles",
            "palliates", "treats", "binds", "causes", "localizes",
            "associates", "presents", "participates", "covaries",
            "interacts", "regulates", "includes", "functions",
            "is_associated_with", "manifests_in", "is_similar_to",
            "acts_on", "is_affected_by", "is_involved_in", "fusion",
            "cooccurrence", "homology", "coexpression", "experiments",
            "database", "textmining", "coexpression_transferred",
            "database_transferred", "experiments_transferred",
            "derives_from", "originates_from", "is_caused_by",
            "presents_as", "is_composed_of"
        }

    @staticmethod
    def load_predictions(tcm_file, compound_file):
        drugs = set()
        if os.path.exists(tcm_file):
            try:
                df = pd.read_csv(tcm_file, sep='\t', usecols=['entity'])
                drugs.update(df['entity'].dropna().unique())
            except Exception as e:
                logger.error(f"加载中药预测失败: {str(e)}")
        if os.path.exists(compound_file):
            try:
                df = pd.read_csv(compound_file, sep='\t', usecols=['entity'])
                drugs.update(df['entity'].dropna().unique())
            except Exception as e:
                logger.error(f"加载化合物预测失败: {str(e)}")
        return list(drugs)

    def is_target_disease(self, entity_id: str) -> bool:
        return any(entity_id.startswith(pfx) for pfx in self.target_disease_prefixes)

    def get_entity_type(self, entity_id):
        if self.is_target_disease(entity_id): return "DiseaseMM"
        for prefix, entity_type in self.entity_prefix_map.items():
            if entity_id.startswith(prefix): return entity_type
        return self.entity_prefix_map["__DEFAULT__"]

    def populate_disease_cache(self):
        for entity in self.index['all_entities']:
            if self.get_entity_type(entity) in ["DiseaseMM", "DiseaseTCM"]:
                self.disease_cache.add(entity)

    def is_disease_entity(self, entity_id):
        return entity_id in self.disease_cache

    def parse_rules(self, rule_file):
        rules = defaultdict(list)
        try:
            with open(rule_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    original_line = line.strip()
                    if not original_line: continue
                    parts = original_line.split('\t')
                    if len(parts) < 4: continue
                    rule_id, rule_length, confidence = parts[0], parts[1], parts[2]
                    rule_str = ' '.join(parts[3:])
                    if ' <= ' not in rule_str: continue
                    head, body = rule_str.split(' <= ', 1)
                    head = head.replace(')', '').strip()
                    head_parts = head.split('(')
                    if len(head_parts) != 2: continue
                    rel = head_parts[0].strip()
                    head_vars = [v.strip() for v in head_parts[1].split(',')]
                    body_atoms = []
                    for pred, var_str in re.findall(r'([a-z_]+)\(([^)]+)\)', body):
                        vars2 = [v.strip() for v in var_str.split(',')]
                        if len(vars2) == 2: body_atoms.append((pred, vars2[0], vars2[1]))
                    rule_data = {
                        'id': rule_id, 'length': rule_length, 'confidence': confidence,
                        'head_rel': rel, 'head_vars': head_vars, 'body_atoms': body_atoms,
                        'raw_rule': original_line
                    }
                    rules[rel].append(rule_data)
        except Exception as e:
            logger.error(f"解析规则异常: {str(e)}")
        return rules

    def is_variable(self, s):
        return s in ['X', 'Y', "Z", "A", "B", "C", "D", "E", "F", "W"]

    def get_possible_objects(self, subject, relation):
        return self.kg_index.get((subject, relation), {}).get('objects', set())

    def get_drug_relations(self, drug):
        if drug in self.drug_relations_cache: return self.drug_relations_cache[drug]
        relations = set()
        if drug in self.index['head_rel']:
            relations = set(self.index['head_rel'][drug].keys())
        self.drug_relations_cache[drug] = relations
        return relations

    def match_rule_body(self, rule, initial_bindings, drug, max_depth=4, max_paths=15):
        if len(rule['body_atoms']) > max_depth: return []
        paths = []
        all_atoms = rule['body_atoms']
        queue = [(initial_bindings.copy(), [drug], set())]
        path_count = 0
        while queue and path_count < max_paths:
            bindings, current_path, completed_atoms = queue.pop(0)
            if len(completed_atoms) == len(all_atoms):
                paths.append((bindings, current_path))
                path_count += 1
                continue
            if len(completed_atoms) >= max_depth: continue
            for idx, (rel, subj, obj) in enumerate(all_atoms):
                if idx in completed_atoms: continue
                subj_val = bindings.get(subj, subj) if self.is_variable(subj) else subj
                obj_val = bindings.get(obj, obj) if self.is_variable(obj) else obj
                if not self.is_variable(subj_val):
                    possible_objects = self.get_possible_objects(subj_val, rel)
                    for o in possible_objects:
                        if self.is_variable(obj_val) or o == obj_val:
                            new_bindings = bindings.copy()
                            if self.is_variable(obj_val): new_bindings[obj_val] = o
                            new_path = current_path.copy()
                            new_path.extend([rel, o])
                            new_completed = completed_atoms.copy()
                            new_completed.add(idx)
                            queue.append((new_bindings, new_path, new_completed))
                            if path_count >= max_paths: break
                elif subj_val in bindings and not self.is_variable(bindings[subj_val]):
                    subj_val_bound = bindings[subj_val]
                    possible_objects = self.get_possible_objects(subj_val_bound, rel)
                    for o in possible_objects:
                        if self.is_variable(obj_val) or o == obj_val:
                            new_bindings = bindings.copy()
                            if self.is_variable(obj_val): new_bindings[obj_val] = o
                            new_path = current_path.copy()
                            new_path.extend([rel, o])
                            new_completed = completed_atoms.copy()
                            new_completed.add(idx)
                            queue.append((new_bindings, new_path, new_completed))
                            if path_count >= max_paths: break
        return paths

    # ================== 核心优化：Score Path (修正版) ==================
    def score_path(self, path, rule_confidence=1.0, is_rule_based=False):
        """
        计算路径得分 (DFS 惩罚修正为 0.2)
        注意：ISI (特异性) 在这里被正式应用
        """
        if len(path) < 3 or (len(path) % 2) != 1:
            return 0.0, {"error": "Invalid path structure"}

        num_relations = (len(path) - 1) // 2
        length_decay_factor = self.get_length_decay(num_relations)

        product = 1.0
        entity_weights = []
        rel_weights = []

        for i in range(0, len(path) - 1, 2):
            if i + 1 >= len(path): break
            rel = path[i + 1]
            rel_weight = self.relation_weights.get(rel, self.relation_weights["__DEFAULT__"])
            rel_weights.append(rel_weight)

            if i + 2 < len(path):
                entity = path[i + 2]
                entity_type = self.get_entity_type(entity)
                base_entity_weight = self.entity_type_weights.get(entity_type, self.entity_type_weights["__DEFAULT__"])

                final_entity_weight = base_entity_weight
                # 【ISI 应用】在这里正式惩罚通用成分
                if entity.startswith("INT"):
                    isi_factor = self.ingredient_specificity.get(entity, 1.0)
                    final_entity_weight *= isi_factor

                entity_weights.append(final_entity_weight)
                product *= rel_weight * final_entity_weight

        geometric_mean = product ** (1.0 / num_relations) if num_relations > 0 else 0.0

        # 【核心修改点 2】DFS Penalty 调整为 0.2
        if is_rule_based:
            effective_confidence = min(1.0, float(rule_confidence))
        else:
            base_score = geometric_mean * length_decay_factor
            dfs_penalty = 0.2  # 0.2 DFS 惩罚
            effective_confidence = min(1.0, base_score * dfs_penalty)

        bonus = 1.0
        path_nodes = set(path[::2])
        if any(ent.startswith("GEE") or ent.startswith("INT") for ent in path_nodes):
            bonus *= 1.2

        exp_rels = {"experiments", "includes", "experiments_transferred"}
        path_rels = set(path[1::2])
        if any(rel in exp_rels for rel in path_rels):
            bonus *= 1.8

        raw_score = geometric_mean * length_decay_factor * effective_confidence * bonus
        final_score = max(0.0, min(raw_score, 1.0))

        score_details = {
            'geometric_mean': geometric_mean,
            'length_decay': length_decay_factor,
            'effective_confidence': effective_confidence,
            'bonus': bonus,
            'final_score': final_score,
            'entity_weights': entity_weights,
            'rel_weights': rel_weights
        }
        return final_score, score_details

    def get_length_decay(self, num_relations):
        if num_relations <= 2:
            return 1.0
        elif num_relations <= 4:
            return 0.9 ** (num_relations - 2)
        else:
            return 0.7 ** (num_relations - 2)

    # ================== DFS 搜索 (VIP策略优化版) ==================
    def find_disease_paths_dfs(self, drug, max_depth=4):
        """
        DFS 搜索优化：
        1. BEAM_WIDTH 扩大到 100
        2. 'includes' 关系享受 VIP 权重 (仅在搜索阶段)，保证成分入选
        3. 搜索阶段不应用 ISI 惩罚 (延迟到 score_path 计算)
        """
        # 【优化1】扩大 Beam
        BEAM_WIDTH = 100
        MAX_PATHS = 500

        stack = [(drug, [drug], [], 1.0)]
        found_paths = []
        path_set = set()

        while stack:
            if len(found_paths) >= MAX_PATHS: break
            current, entity_path, rel_path, current_cum_score = stack.pop()

            if self.is_target_disease(current):
                path_tuple = (tuple(entity_path), tuple(rel_path))
                if path_tuple not in path_set:
                    full_path = []
                    for i in range(len(entity_path)):
                        full_path.append(entity_path[i])
                        if i < len(rel_path): full_path.append(rel_path[i])

                    # 统一调用 score_path 计算最终分数 (这里会应用 ISI)
                    final_score, score_details = self.score_path(full_path, is_rule_based=False)
                    found_paths.append((full_path, "BFS_PATH", final_score, score_details))
                    path_set.add(path_tuple)
                continue

            if len(rel_path) >= max_depth: continue
            if current not in self.index['head_rel']: continue

            candidates = []
            for rel, tails in self.index['head_rel'][current].items():
                if rel not in self.relation_whitelist: continue

                # 【优化2】VIP 策略：给 'includes' 关系发 VIP 卡
                if rel == "includes":
                    rel_weight = 2.0  # 临时高权重，只为挤进 Beam
                else:
                    rel_weight = self.relation_weights.get(rel, self.relation_weights["__DEFAULT__"])

                for tail in tails:
                    if tail in entity_path: continue
                    entity_type = self.get_entity_type(tail)

                    # 【优化3】延迟 ISI：搜索阶段不乘 ISI，只用基础权重
                    # 这样通用成分也能暂时保留
                    ent_weight = self.entity_type_weights.get(entity_type, 0.7)

                    step_score = rel_weight * ent_weight
                    new_cum_score = current_cum_score * step_score
                    if new_cum_score < 1e-4: continue
                    candidates.append((tail, rel, new_cum_score))

            # 先按分数(x[2])降序，分数相同按实体名(x[0])降序
            candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
            top_candidates = candidates[:BEAM_WIDTH]
            for tail, rel, score in top_candidates:
                new_entity_path = entity_path + [tail]
                new_rel_path = rel_path + [rel]
                stack.append((tail, new_entity_path, new_rel_path, score))

        # 先按分数(x[2])降序，分数相同按路径字符串(str(x[0]))降序
        found_paths.sort(key=lambda x: (x[2], str(x[0])), reverse=True)
        return found_paths[:200]

    def find_rule_based_path(self, drug):
        rule_paths = []
        if "candidate_for" not in self.rules or not self.rules["candidate_for"]: return []

        drug_relations = self.get_drug_relations(drug)
        # 修复变量名错误逻辑
        candidate_rules = []
        for rule in self.rules["candidate_for"]:
            if not rule['body_atoms']: continue
            if rule['body_atoms'][0][0] in drug_relations:
                candidate_rules.append(rule)

        for rule in candidate_rules:
            head_x, head_y = rule['head_vars']
            bindings0 = {head_x: drug}
            paths = self.match_rule_body(rule, bindings0, drug, max_depth=4, max_paths=15)
            if paths:
                for bind_i, cur_path in paths:
                    tail_entity = head_y
                    if self.is_variable(tail_entity):
                        tail_entity = bind_i.get(tail_entity, tail_entity)
                    if not isinstance(tail_entity, str) or not self.is_target_disease(tail_entity):
                        continue
                    new_path = cur_path + [rule['head_rel'], tail_entity]
                    confidence = float(rule.get('confidence', 0.5))
                    path_score, score_details = self.score_path(new_path, confidence, is_rule_based=True)
                    rule_paths.append((new_path, rule['raw_rule'], path_score, score_details))

        rule_paths.sort(key=lambda x: x[2], reverse=True)
        return rule_paths[:15]

    def get_paths_for_drug(self, drug, need_top_k=15):
        # 1. Rules
        r_paths = self.find_rule_based_path(drug)
        # 2. DFS
        d_paths = self.find_disease_paths_dfs(drug, max_depth=4)

        # 3. Merge (软竞争 + 智能去重)
        uniq = {}
        # 先放入 Rule 路径 (Rule标签优先)
        for p in r_paths:
            sig = tuple(p[0])
            if sig not in uniq or p[2] > uniq[sig][2]:
                uniq[sig] = p

        # 再尝试放入 DFS 路径 (只要坑被占了就不抢)
        for p in d_paths:
            sig = tuple(p[0])
            if sig not in uniq:
                uniq[sig] = p

        final = list(uniq.values())
        # 【修复】双重排序：先按分数(x[2])降序，再按路径字符串(str(x[0]))降序
        # 这样能保证同分路径的顺序永远一致，且与 05 脚本对齐
        final.sort(key=lambda x: (x[2], str(x[0])), reverse=True)
        return final[:need_top_k]

    def clean_cache_dir(self, cache_dir, max_age_hours=24):
        now = time.time()
        removed_count = 0
        try:
            for filename in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, filename)
                if not os.path.isfile(file_path): continue
                file_age = now - os.path.getmtime(file_path)
                if file_age > max_age_hours * 3600:
                    try:
                        os.remove(file_path);
                        removed_count += 1
                    except Exception as e:
                        logger.error(f"Del err: {e}")
        except Exception:
            pass
        return removed_count


def process_one_drug(explainer: PathExplainer, drug: str, cache_dir: str):
    explainer.check_memory_usage()
    cache_file = os.path.join(cache_dir, f"{drug}_cache.pkl")
    rule_paths = None
    if os.path.exists(cache_file):
        try:
            with explainer.cache_lock:
                with open(cache_file, 'rb') as f_cache: rule_paths = pickle.load(f_cache)
        except Exception:
            pass
    if not rule_paths:
        rule_paths = explainer.get_paths_for_drug(drug, need_top_k=15)
        try:
            with explainer.cache_lock:
                with open(cache_file, 'wb') as f_cache: pickle.dump(rule_paths, f_cache)
        except Exception:
            pass
    return drug, rule_paths


if __name__ == "__main__":
    logger.info("启动路径解释器（包含成分特异性优化 ISI + 统一评分逻辑）")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    cache_dir = os.path.join(project_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    kg_file = os.path.join(project_dir, "data", "final_unique_merged_triples.txt")
    rule_dir = os.path.join(project_dir, "anyburl", "rules")
    compound_file = os.path.join(project_dir, "results", "predictions", "top200_western.tsv")
    tcm_file = os.path.join(project_dir, "results", "predictions", "top100_tcm.tsv")
    output_file = os.path.join(project_dir, "explanations", "DM_06_INT_to100_full_40paths_optimized.tsv")
    summary_file = os.path.join(project_dir, "explanations", "DM_06_INT_to100_40path_summary_optimized.tsv")

    try:
        explainer = PathExplainer(kg_file, rule_dir, disease_prefixes=["DM06"])
    except Exception as e:
        logger.error(f"Init error: {str(e)}");
        exit(1)

    drugs = PathExplainer.load_predictions(tcm_file, compound_file)
    logger.info(f"找到 {len(drugs)} 种药物")

    MAX_WORKERS = min(8, max(2, os.cpu_count() or 4))
    results = {}
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, drug in enumerate(drugs):
            if i > 0 and i % explainer.CACHE_CLEAN_INTERVAL == 0:
                explainer.clean_cache_dir(cache_dir)
                gc.collect()
            futures.append(executor.submit(process_one_drug, explainer, drug, cache_dir))

        for f in tqdm(as_completed(futures), total=len(futures), desc="处理药物(并行)"):
            try:
                drug, rule_paths = f.result()
                results[drug] = rule_paths
            except Exception as e:
                logger.error(f"并行任务异常：{e}")
                continue

    with open(output_file, 'w', newline='', encoding='utf-8') as f, \
            open(summary_file, 'w', newline='', encoding='utf-8') as summary_f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["Drug", "Path", "Rule", "Score"])
        summary_writer = csv.writer(summary_f, delimiter='\t')
        summary_writer.writerow(["Drug", "PathType", "MaxScore"])

        for drug in drugs:
            rule_paths = results.get(drug, [])
            if not rule_paths: continue
            path_type = "DFS" if rule_paths[0][1] == "BFS_PATH" else "元规则"
            max_score = rule_paths[0][2]
            summary_writer.writerow([drug, path_type, f"{max_score:.4f}"])
            for path_info, rule_info, score, _ in rule_paths[:10]:
                path_str = " → ".join(path_info) if isinstance(path_info, list) else str(path_info)
                writer.writerow([drug, path_str, rule_info, f"{score:.4f}"])

    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
        except:
            pass
    logger.info(f"处理完成! 路径详情保存在: {output_file}")