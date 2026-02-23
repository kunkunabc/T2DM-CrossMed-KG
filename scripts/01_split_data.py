import os
import pandas as pd
import networkx as nx
from sklearn.model_selection import StratifiedShuffleSplit
from ruamel.yaml import YAML
import numpy as np

# === 配置与路径 ===
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_path = os.path.join(project_root, "configs", "default.yaml")

# 读取 YAML 配置
yaml_loader = YAML(typ="safe", pure=True)
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml_loader.load(f)

data_path = os.path.join(project_root, cfg["data"]["triplets_path"])
output_dir = os.path.join(project_root, "splits")
os.makedirs(output_dir, exist_ok=True)

# 划分比例
TEST_SIZE = cfg["splits"]["test_size"]
VALID_SIZE = cfg["splits"]["valid_size"]
RANDOM_SEED = cfg.get("seed", 42)

# 1. 读取原始三元组并保留原始行号
print(f"读取原始三元组: {data_path}")
df = pd.read_csv(data_path, header=None, names=["head", "relation", "tail"], dtype=str)
df = df.fillna('')  # 填充可能的NaN值为空字符串
df['source_line'] = df.index  # 保留原始行号
print(f"原始三元组总数: {len(df)}")


# 2. 检测并记录空实体
def detect_empty_entities(row):
    """检测并记录空实体行"""
    head_str = str(row['head']) if pd.notna(row['head']) else ""
    tail_str = str(row['tail']) if pd.notna(row['tail']) else ""

    if head_str.strip() == '':
        return f"空头实体:行{row.name}"
    if tail_str.strip() == '':
        return f"空尾实体:行{row.name}"
    return None


# 确保所有值都是字符串
df['head'] = df['head'].astype(str)
df['relation'] = df['relation'].astype(str)
df['tail'] = df['tail'].astype(str)

empty_mask = (df['head'].str.strip() == '') | (df['tail'].str.strip() == '')
if empty_mask.any():
    empty_entities = df[empty_mask]
    error_log = os.path.join(output_dir, 'empty_entities.log')
    with open(error_log, 'w') as f:
        for idx, row in empty_entities.iterrows():
            f.write(
                f"Error in source line {row['source_line']}: ['{row['head']}', '{row['relation']}', '{row['tail']}']\n")
    print(f"⚠️ 发现 {len(empty_entities)} 行空实体三元组，已保存至: {error_log}")

    # 过滤掉无效行
    df = df[~empty_mask].reset_index(drop=True)
    print(f"移除空实体后三元组数: {len(df)}")

# 3. 连通性过滤（保留主连通分量）
print("进行连通性过滤...")
G = nx.Graph()

for idx, row in df.iterrows():
    head = str(row['head']).strip()
    tail = str(row['tail']).strip()
    if head and tail:  # 确保实体非空
        G.add_edge(head, tail, line=row['source_line'])

components = list(nx.connected_components(G))
giant = max(components, key=len)


# 创建一个映射函数检查实体是否在巨型组件中
def in_giant(row):
    head_str = str(row['head']).strip()
    tail_str = str(row['tail']).strip()
    return head_str in giant and tail_str in giant


df['in_giant'] = df.apply(in_giant, axis=1)
df = df[df['in_giant']].drop(columns=['in_giant']).reset_index(drop=True)
print(f"主连通分量三元组数: {len(df)}")

# 4. 分层抽样
print("进行分层抽样...")
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
idx_train_val, idx_test = next(sss1.split(df, df['relation']))
df_train_val = df.iloc[idx_train_val].reset_index(drop=True)
df_test = df.iloc[idx_test].reset_index(drop=True)

valid_frac = VALID_SIZE / (1 - TEST_SIZE)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=valid_frac, random_state=RANDOM_SEED)
idx_train, idx_valid = next(sss2.split(df_train_val, df_train_val['relation']))
df_train = df_train_val.iloc[idx_train].reset_index(drop=True)
df_valid = df_train_val.iloc[idx_valid].reset_index(drop=True)


# 5. 修改后的覆盖性检查函数（处理类型问题）
def check_coverage(df_train, df_sub, subset_name):
    """检查实体覆盖并返回缺失报告"""
    train_ents = set(df_train['head']) | set(df_train['tail'])
    train_rels = set(df_train['relation'])

    coverage_errors = []

    for idx, row in df_sub.iterrows():
        source_line = row['source_line']
        try:
            # 确保所有值都是字符串
            head = str(row['head']).strip()
            rel = str(row['relation']).strip()
            tail = str(row['tail']).strip()

            # 空实体检查
            if not head or not tail:
                coverage_errors.append(f"Error in source line {source_line}: ['{head}', '{rel}', '{tail}'] - 空实体")
                continue

            # 实体和关系覆盖检查
            if head not in train_ents:
                coverage_errors.append(
                    f"Error in source line {source_line}: ['{head}', '{rel}', '{tail}'] - 缺失头实体'{head}'")
            if tail not in train_ents:
                coverage_errors.append(
                    f"Error in source line {source_line}: ['{head}', '{rel}', '{tail}'] - 缺失尾实体'{tail}'")
            if rel not in train_rels:
                coverage_errors.append(
                    f"Error in source line {source_line}: ['{head}', '{rel}', '{tail}'] - 缺失关系'{rel}'")
        except Exception as e:
            print(f"处理行 {source_line} 时出错: {e}")
            coverage_errors.append(f"Error in source line {source_line}: 处理错误 - {str(e)}")

    return coverage_errors


# 6. 覆盖性修复
print("进行覆盖性检查和修复...")
all_errors = []

for iteration in range(1, 4):  # 最多3次迭代
    print(f"\n=== 覆盖检查迭代 #{iteration} ===")
    errors_valid = check_coverage(df_train, df_valid, 'valid')
    errors_test = check_coverage(df_train, df_test, 'test')

    current_errors = errors_valid + errors_test
    print(f"发现 {len(current_errors)} 个覆盖问题")

    if not current_errors:
        print("✓ 所有覆盖问题已解决")
        break

    # 记录错误
    all_errors.extend(current_errors)

    # 提取问题行号
    problem_lines = set()
    for err in current_errors:
        # 解析原始行号: "Error in source line 1234: ..."
        parts = err.split(":")
        if len(parts) > 0:
            line_str = parts[0].split(" ")[-1]
            if line_str.isdigit():
                problem_lines.add(int(line_str))

    if not problem_lines:
        print("⚠️ 无法解析错误信息中的行号")
        break

    # 收集问题三元组
    problem_triplets = df[df['source_line'].isin(problem_lines)]

    print(f"将 {len(problem_triplets)} 条问题三元组移到训练集")

    # 从原始验证/测试集移出
    valid_mask = df_valid['source_line'].isin(problem_lines)
    test_mask = df_test['source_line'].isin(problem_lines)

    to_remove_valid = df_valid[valid_mask]
    to_remove_test = df_test[test_mask]

    df_valid = df_valid[~valid_mask].reset_index(drop=True)
    df_test = df_test[~test_mask].reset_index(drop=True)

    # 添加到训练集
    df_train = pd.concat([df_train, to_remove_valid, to_remove_test], ignore_index=True)

    print(f"更新后大小: 训练集={len(df_train)}, 验证集={len(df_valid)}, 测试集={len(df_test)}")

# 7. 最终覆盖检查
print("\n=== 最终覆盖检查 ===")
final_errors = []
final_errors.extend(check_coverage(df_train, df_valid, 'valid'))
final_errors.extend(check_coverage(df_train, df_test, 'test'))
all_errors.extend(final_errors)

if final_errors:
    print(f"⚠️ 发现 {len(final_errors)} 个最终覆盖问题")
    for err in final_errors[:10]:  # 显示前10个错误
        print(err)
else:
    print("✓ 所有实体和关系均已正确覆盖")

# 8. 保存结果
print("\n保存划分结果...")
# 保存之前，去掉source_line列
df_train.drop(columns=['source_line']).to_csv(os.path.join(output_dir, 'train.tsv'), sep='\t', header=False,
                                              index=False)
df_valid.drop(columns=['source_line']).to_csv(os.path.join(output_dir, 'valid.tsv'), sep='\t', header=False,
                                              index=False)
df_test.drop(columns=['source_line']).to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t', header=False, index=False)

# 保存错误日志
if all_errors:
    error_log = os.path.join(output_dir, 'coverage_errors.log')
    with open(error_log, 'w') as f:
        f.write("\n".join(all_errors))
    print(f"⚠️ 共发现 {len(all_errors)} 个覆盖问题，已保存至: {error_log}")

# 统计摘要
original_count = len(df)
train_count = len(df_train)
valid_count = len(df_valid)
test_count = len(df_test)
dropped = original_count - (train_count + valid_count + test_count)

print("\n✔ 数据集划分完成")
print(f"最终大小: 训练集={train_count}, 验证集={valid_count}, 测试集={test_count}")
print(f"丢弃行数: {dropped} (主要是空实体和未进入主连通分量的三元组)")
print(f"总计: {train_count + valid_count + test_count} (原始: {original_count})")
print(f"文件保存在: {output_dir}")