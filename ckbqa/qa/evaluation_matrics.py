def get_metrics(real_entities, pred_entities):
    real_entities_set = set(real_entities)
    pred_entities_set = set(pred_entities)
    TP = len(real_entities_set & pred_entities_set)  # 负例 TP
    # FN = len(pred_entities - real_entities)  # 预测出的负例, FN
    # FP = len(real_entities - pred_entities)  # 没有预测出的正例
    # TP+FP = real_entities_set
    # TP+FN = pred_entities_set
    precision = TP / len(pred_entities)
    recall = TP / len(real_entities_set)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
