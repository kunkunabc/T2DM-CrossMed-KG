# python 03_predict_drugs.py

import os
from tqdm import tqdm
import torch
from heapq import heappush, heappop
from pykeen.triples import TriplesFactory


def batch_score(model, triples, batch_size=2048):
    """分批打分并返回所有分数 CPU 张量列表"""
    scores = []
    for i in range(0, len(triples), batch_size):
        batch = triples[i:i + batch_size]
        with torch.no_grad():
            s = model.score_hrt(batch).squeeze().cpu()
        scores.append(s)
        del batch, s
    return torch.cat(scores)


def stream_extract_topk_fullinfo(src_csv, top_k, out_csv):
    """
    低内存模式流式提取 TopK，保留完整字段
    输出文件列：query,relation,direction,entity,score,rank
    """
    heap = []  # min-heap 中存 (score, parts)
    seen = set()  # 已收录实体
    with open(src_csv, 'r', encoding='utf-8') as fin:
        header = fin.readline()  # skip header
        for line in tqdm(fin, desc=f"提取 Top{top_k}", unit="行"):
            parts = line.rstrip('\n').split(',')
            if len(parts) != 6:
                continue
            query, relation, direction, entity, score_str, rank = parts
            if entity in seen:
                continue
            try:
                score = float(score_str)
            except ValueError:
                continue
            if len(heap) < top_k:
                heappush(heap, (score, parts))
                seen.add(entity)
            elif score > heap[0][0]:
                _, removed = heappop(heap)
                seen.remove(removed[3])
                heappush(heap, (score, parts))
                seen.add(entity)

    # 排序并写出
    top = sorted(heap, key=lambda x: x[0], reverse=True)
    with open(out_csv, 'w', encoding='utf-8') as fout:
        fout.write("query,relation,direction,entity,score,rank\n")
        for _, parts in top:
            fout.write(','.join(parts) + '\n')
    print(f"✅ Top{top_k} 提取完毕，保存至: {out_csv}")


def predict_for_set(label, drug_list, tf, model, device, output_dir):
    """对一组候选 drug_list 预测并写入 incremental CSV，返回该 CSV 路径"""
    ids = torch.tensor([tf.entity_to_id[e] for e in drug_list], device=device)
    num = len(ids)
    output_file = os.path.join(output_dir, f"predicted_drugs_{label}.csv")
    # 写表头
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("query,relation,direction,entity,score,rank\n")

    # 包含所有6种关系
    forward_rels = ["binds", "downregulates", "upregulates", "functions", "is_associated_with"]

    for gene in tqdm(diabetes_genes, desc=f'[{label}] 基因', unit='gene'):
        if gene not in tf.entity_to_id:
            continue
        gid = tf.entity_to_id[gene]

        for rel in forward_rels:
            if rel not in tf.relation_to_id:
                continue
            rid = tf.relation_to_id[rel]
            direction = "h→t"  # 统一方向标记

            # === 根据关系类型动态构建三元组 ===
            if rel == "is_associated_with":
                # 中药候选集特殊处理："中药：关联：基因" (h:中药, r:关联, t:基因)
                if label == 'tcm':
                    h = ids
                    r = torch.full((num,), rid, device=device)
                    t = torch.full((num,), gid, device=device)
                    triples = torch.stack([h, r, t], dim=1)
                # 其他候选集："基因：关联：化合物" (h:基因, r:关联, t:化合物)
                else:
                    h = torch.full((num,), gid, device=device)
                    r = torch.full((num,), rid, device=device)
                    t = ids
                    triples = torch.stack([h, r, t], dim=1)
            else:
                # 其他关系统一格式："化合物：关系：基因" (h:化合物, r:关系, t:基因)
                h = ids
                r = torch.full((num,), rid, device=device)
                t = torch.full((num,), gid, device=device)
                triples = torch.stack([h, r, t], dim=1)
            # ==============================

            # 分批打分
            scores = batch_score(model, triples)
            sorted_scores, idxs = torch.sort(scores, descending=True)
            sorted_scores = sorted_scores.numpy()
            idxs = idxs.numpy().astype(int)

            # 逐行写入
            with open(output_file, 'a', encoding='utf-8') as f:
                for rank, i in enumerate(idxs, 1):
                    f.write(f"{gene},{rel},{direction},{drug_list[i]},{sorted_scores[rank - 1]},{rank}\n")

            # 释放资源
            del triples, scores, sorted_scores, idxs
            torch.cuda.empty_cache()

    print(f"✔ [{label}] 预测完成，结果保存在: {output_file}")
    return output_file


if __name__ == "__main__":
    # ——— 路径配置 —————————————————————————————
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    train_path = os.path.join(project_dir, "splits", "train.tsv")
    gene_path = os.path.join(project_dir, "data", "DM_Gene.txt")
    paths = {
        'all': os.path.join(project_dir, "data", "All_Drug.txt"),
        'tcm': os.path.join(project_dir, "data", "TCM_Drug.txt"),
        'western': os.path.join(project_dir, "data", "MM_Drug.txt"),
    }
    output_dir = os.path.join(project_dir, "results", "predictions")
    os.makedirs(output_dir, exist_ok=True)

    # ——— 加载 TriplesFactory & 模型 —————————————————
    tf = TriplesFactory.from_path(train_path, create_inverse_triples=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(os.path.join(project_dir, "results", "RotatE", "trained_model.pkl"),
                       map_location=device)
    model.to(device).eval()

    # ——— 读取糖尿病基因 ———————————————————————
    with open(gene_path, 'r', encoding='utf-8') as f:
        diabetes_genes = [l.strip() for l in f if l.strip()]

    # ——— 依次预测三组候选集 —————————————————————
    outputs = {}
    for label, path in paths.items():
        with open(path, 'r', encoding='utf-8') as f:
            drugs = [l.strip() for l in f
                     if l.strip() and l.strip() in tf.entity_to_id]
        if not drugs:
            print(f"⚠️ [{label}] 候选实体为空，已跳过")
            continue
        outputs[label] = predict_for_set(label, drugs, tf, model, device, output_dir)

    # ——— 流式提取 TopK（保留完整字段） —————————————————
    topk_config = {'all': 200, 'tcm': 100, 'western': 200}
    for label, src in outputs.items():
        topk = topk_config[label]
        top_csv = os.path.join(output_dir, f"top{topk}_{label}.csv")
        stream_extract_topk_fullinfo(src, topk, top_csv)

    print("🎉 所有任务完成！")