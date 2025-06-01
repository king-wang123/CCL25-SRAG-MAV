import json
import difflib
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import re


def parse_output_to_quadruples(output_text):
    """
    将输出文本解析为四元组列表
    格式: "Target | Argument | Targeted Group | Hateful [SEP] ..."
    """
    quadruples = []
    
    # 按[SEP]分割多个四元组
    for quad_text in re.split(r'\s*\[SEP\]\s*', output_text):
        # 去掉末尾的[END]
        quad_text = re.sub(r'\s*\[END\]\s*$', '', quad_text.strip())
        
        # 如果文本为空则跳过
        if not quad_text:
            continue
            
        # 按|分割四元组的各个元素
        elements = [elem.strip() for elem in quad_text.split('|')]
        
        # 确保有4个元素
        if len(elements) == 4:
            quadruples.append({
                'Target': elements[0],
                'Argument': elements[1],
                'Targeted_Group': elements[2],
                'Hateful': elements[3]
            })
    
    return quadruples


def calculate_similarity(pred_text, gold_text):
    """
    使用difflib.SequenceMatcher计算两个字符串的相似度
    """
    seq_matcher = difflib.SequenceMatcher(None, pred_text, gold_text)
    similarity = seq_matcher.ratio()
    return similarity


def is_hard_match(pred_quad, gold_quad):
    """
    判断预测四元组和标准答案是否硬匹配
    硬匹配：四元组的每个元素完全一致
    """
    return (pred_quad['Target'] == gold_quad['Target'] and
            pred_quad['Argument'] == gold_quad['Argument'] and
            pred_quad['Targeted_Group'] == gold_quad['Targeted_Group'] and
            pred_quad['Hateful'] == gold_quad['Hateful'])


def is_soft_match(pred_quad, gold_quad):
    """
    判断预测四元组和标准答案是否软匹配
    软匹配：Targeted_Group和Hateful完全一致，Target和Argument相似度>0.5
    """
    # 必须完全匹配的元素
    if (pred_quad['Targeted_Group'] != gold_quad['Targeted_Group'] or
            pred_quad['Hateful'] != gold_quad['Hateful']):
        return False
    
    # 计算Target的相似度
    target_similarity = calculate_similarity(pred_quad['Target'], gold_quad['Target'])
    
    # 计算Argument的相似度
    argument_similarity = calculate_similarity(pred_quad['Argument'], gold_quad['Argument'])
    
    # 如果相似度都超过0.5则匹配成功
    return target_similarity > 0.5 and argument_similarity > 0.5


def evaluate_predictions(gold_data, pred_data):
    """
    评估预测结果，计算硬匹配和软匹配的F1分数
    """
    # 创建ID到四元组列表的映射
    gold_quads_by_id = {}
    pred_quads_by_id = {}
    
    # 解析标准答案
    for item in gold_data:
        item_id = item['id']
        quads = parse_output_to_quadruples(item['output'])
        gold_quads_by_id[item_id] = quads
    
    # 解析预测结果
    for item in pred_data:
        item_id = item['id']
        quads = parse_output_to_quadruples(item['output'])
        pred_quads_by_id[item_id] = quads
    
    # 收集每个示例的硬匹配和软匹配结果，用于计算总体F1
    all_hard_tp, all_hard_fp, all_hard_fn = 0, 0, 0
    all_soft_tp, all_soft_fp, all_soft_fn = 0, 0, 0
    
    # 对每个示例进行评估
    for item_id in gold_quads_by_id:
        gold_quads = gold_quads_by_id.get(item_id, [])
        pred_quads = pred_quads_by_id.get(item_id, [])
        
        # 硬匹配评估
        hard_matched_pred = set()
        hard_matched_gold = set()
        
        for i, pred_quad in enumerate(pred_quads):
            for j, gold_quad in enumerate(gold_quads):
                if j in hard_matched_gold:
                    continue
                if is_hard_match(pred_quad, gold_quad):
                    hard_matched_pred.add(i)
                    hard_matched_gold.add(j)
                    break
        
        hard_tp = len(hard_matched_pred)
        hard_fp = len(pred_quads) - hard_tp
        hard_fn = len(gold_quads) - len(hard_matched_gold)
        
        all_hard_tp += hard_tp
        all_hard_fp += hard_fp
        all_hard_fn += hard_fn
        
        # 软匹配评估
        soft_matched_pred = set()
        soft_matched_gold = set()
        
        for i, pred_quad in enumerate(pred_quads):
            for j, gold_quad in enumerate(gold_quads):
                if j in soft_matched_gold:
                    continue
                if is_soft_match(pred_quad, gold_quad):
                    soft_matched_pred.add(i)
                    soft_matched_gold.add(j)
                    break
        
        soft_tp = len(soft_matched_pred)
        soft_fp = len(pred_quads) - soft_tp
        soft_fn = len(gold_quads) - len(soft_matched_gold)
        
        all_soft_tp += soft_tp
        all_soft_fp += soft_fp
        all_soft_fn += soft_fn
    
    # 计算硬匹配的精确率、召回率和F1分数
    hard_precision = all_hard_tp / (all_hard_tp + all_hard_fp) if (all_hard_tp + all_hard_fp) > 0 else 0
    hard_recall = all_hard_tp / (all_hard_tp + all_hard_fn) if (all_hard_tp + all_hard_fn) > 0 else 0
    hard_f1 = 2 * hard_precision * hard_recall / (hard_precision + hard_recall) if (hard_precision + hard_recall) > 0 else 0
    
    # 计算软匹配的精确率、召回率和F1分数
    soft_precision = all_soft_tp / (all_soft_tp + all_soft_fp) if (all_soft_tp + all_soft_fp) > 0 else 0
    soft_recall = all_soft_tp / (all_soft_tp + all_soft_fn) if (all_soft_tp + all_soft_fn) > 0 else 0
    soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0
    
    # 计算平均F1分数
    avg_f1 = (hard_f1 + soft_f1) / 2
    
    return {
        'hard_precision': hard_precision,
        'hard_recall': hard_recall,
        'hard_f1': hard_f1,
        'soft_precision': soft_precision,
        'soft_recall': soft_recall,
        'soft_f1': soft_f1,
        'avg_f1': avg_f1
    }

def main(gold_file, pred_file):
    # 读取测试集和预测结果
    try:
        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        
        # 评估结果
        results = evaluate_predictions(gold_data, pred_data)
        
        # 输出结果
        print(f"Hard Score: {results['hard_f1']:.4f}")
        print(f"Soft Score: {results['soft_f1']:.4f}")
        print(f"Score: {results['avg_f1']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估四元组抽取任务的性能')
    parser.add_argument('--gold', type=str, default='../data/test_split.json', help='标准答案文件路径')
    parser.add_argument('--pred', type=str, help='预测结果文件路径')
    
    args = parser.parse_args()
    main(args.gold, args.pred)