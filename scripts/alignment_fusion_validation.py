import pandas as pd


def validate_entity_alignment():
    print("========================================")
    print("开始实体对齐分层抽样与自动跨库溯源...")
    print("========================================\n")

    # 严格根据你提供的文件结构、Sheet名和列名进行配置
    configs = {
        'Gene': {
            'file': 'Gene.xlsx',
            'unified_sheet': '融合后共26020',
            'unified_name_col': '基因',
            'unified_id_col': '自定义id',
            'source_sheets': [
                'Hetionet去重后19144',
                'TCMBank去重后12320',
                'Symmap去重后19028',
                'TTD去重后2613',
                'STRING去重清洗后19425'
            ],
            'source_name_col': '基因',
            'sample_size': 40
        },
        'DiseaseMM': {
            'file': 'DiseaseMM.xlsx',
            'unified_sheet': '补充糖尿病实体分级名单后38346种',
            'unified_name_col': '疾病MM',
            'unified_id_col': '自定义id',
            'source_sheets': [
                'Hetionet中136种',
                'TCMBank中29499种',
                'Symmap中14426种',
                'TTD中2343种'
            ],
            'source_name_col': '疾病MM',
            'sample_size': 30
        },
        'Herb': {
            'file': 'Herb.xlsx',
            'unified_sheet': '融合去重后共7159种',
            'unified_name_col': '中药',
            'unified_id_col': '自定义id',
            'source_sheets': [
                'TCMBank去重后共6733，无中药名称的结合TCM_id',
                'Symmap共698条'
            ],
            'source_name_col': '中药',
            'sample_size': 30
        }
    }

    validation_results = []

    for entity_type, config in configs.items():
        print(f"-> 正在处理实体类别: {entity_type} ...")
        try:
            # 1. 加载主 Excel 文件
            xls = pd.ExcelFile(config['file'])

            # 2. 读取总表并进行随机抽样
            unified_df = pd.read_excel(xls, sheet_name=config['unified_sheet'])

            # 容错处理：如果列名有微小差异，默认回退到第1列和第2列
            cols = unified_df.columns.tolist()
            name_col = config['unified_name_col'] if config['unified_name_col'] in cols else cols[0]
            id_col = config['unified_id_col'] if config['unified_id_col'] in cols else cols[1]

            sampled_df = unified_df.sample(n=config['sample_size'], random_state=42)

            # 3. 预加载所有子表数据，转化为 Set 加速后续查询
            source_data = {}
            for source_sheet in config['source_sheets']:
                df_source = pd.read_excel(xls, sheet_name=source_sheet)
                s_name_col = config['source_name_col'] if config['source_name_col'] in df_source.columns else \
                df_source.columns[0]
                # 清理首尾空格并转为字符串
                source_data[source_sheet] = set(df_source[s_name_col].astype(str).str.strip())

            # 4. 遍历抽样数据进行跨库自动溯源
            for _, row in sampled_df.iterrows():
                entity_name = str(row[name_col]).strip()
                entity_id = row[id_col]

                found_in_dbs = []
                for source_sheet, name_set in source_data.items():
                    if entity_name in name_set:
                        # 提取干净的数据库名称 (例如将 "Hetionet去重后19144" 截取为 "Hetionet")
                        db_name = source_sheet.split('去')[0].split('中')[0].split('共')[0]
                        found_in_dbs.append(db_name)

                validation_results.append({
                    '实体类型 (Type)': entity_type,
                    '实体名称 (Name)': entity_name,
                    '自定义ID (ID)': entity_id,
                    '跨库溯源发现 (Found_in_Sources)': " | ".join(found_in_dbs),
                    '人工核对: 跨库指代是否100%一致? (Yes/No)': '',
                    '备注 (Remarks)': ''
                })

            print(f"   √ 成功抽取并溯源 {config['sample_size']} 个 {entity_type} 实体。\n")

        except Exception as e:
            print(f"   x 处理 {entity_type} 时出错: {e}\n")

    # 5. 导出实体对齐结果
    if validation_results:
        result_df = pd.DataFrame(validation_results)
        result_df.to_excel('Validation_Entity_Alignment_Detailed.xlsx', index=False)
        print(">>> 实体对齐抽样完成！已生成: Validation_Entity_Alignment_Detailed.xlsx\n")


def validate_relation_fusion():
    print("========================================")
    print("开始关系融合抽样...")
    print("========================================\n")
    try:
        triples_df = pd.read_csv('final_unique_merged_triples.csv')

        # 验证 True Positive：成功融合的 expresses
        tp_samples = triples_df[triples_df['Relation'] == 'expresses'].sample(n=30, random_state=42)
        tp_samples['Manual_Check_Reasonable_Merge? (Yes/No)'] = ''
        tp_samples['Remarks'] = '确认 upregulate 并入 expresses 语义合理'
        tp_samples.to_excel('Validation_Relation_Merge_TP(expresses).xlsx', index=False)
        print("   √ True Positive 关系抽样完成 (expresses)。")

        # 验证 True Negative：拦截融合的 interacts
        tn_samples = triples_df[triples_df['Relation'] == 'interacts'].sample(n=30, random_state=42)
        tn_samples['Manual_Check_Should_Keep_Separate? (Yes/No)'] = ''
        tn_samples['Remarks'] = '确认保持独立于 experiments 的必要性'
        tn_samples.to_excel('Validation_Relation_Merge_TN(interacts).xlsx', index=False)
        print("   √ True Negative 关系抽样完成 (interacts)。\n")

    except Exception as e:
        print(f"   x 处理关系融合抽样时出错: {e}。请确认目录下有 final_unique_merged_triples.csv\n")


if __name__ == '__main__':
    validate_entity_alignment()
    validate_relation_fusion()