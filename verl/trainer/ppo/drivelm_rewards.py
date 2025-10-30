import warnings

# BLEU
def bleu_score(solution_str, ground_truth):
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        refs = [ground_truth.split()]
        hypo = solution_str.split()
        smoothie = SmoothingFunction().method3
        score = sentence_bleu(refs, hypo, smoothing_function=smoothie)
        return float(score)
    except ImportError:
        warnings.warn("nltk not installed, BLEU will be 0.")
        return 0.0
    except Exception:
        return 0.0

# ROUGE-L
def rouge_l_score(solution_str, ground_truth):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        score = scorer.score(ground_truth, solution_str)["rougeL"].fmeasure
        return float(score)
    except ImportError:
        warnings.warn("rouge-score not installed, ROUGE-L will be 0.")
        return 0.0
    except Exception:
        return 0.0

# CIDEr
def cider_score(solution_str, ground_truth):
    try:
        from pycocoevalcap.cider.cider import Cider
        cider = Cider()
        gts = {0: [{"caption": ground_truth}]}
        res = {0: [{"caption": solution_str}]}
        score, _ = cider.compute_score(gts, res)
        return float(score)
    except ImportError:
        warnings.warn("pycocoevalcap cider not installed, CIDEr will be 0.")
        return 0.0
    except Exception:
        return 0.0

# SPICE
def spice_score(solution_str, ground_truth):
    try:
        import evaluate
        spice = evaluate.load("spice")
        gts = {0: [{"caption": ground_truth}]}
        res = {0: [{"caption": solution_str}]}
        score = spice.compute(predictions=[res], references=[[gts]])['score']
        return float(score)
    except ImportError:
        warnings.warn("evaluate not imported correctly, or spice function is not used correctly, SPICE will be 0.")
        return 0.0
    except Exception:
        return 0.0

# 总奖励函数 - 传入参数与verl一致
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    生成总reward和单项reward的dict
    """
    metrics = {
        "BLEU": bleu_score(solution_str, ground_truth),
        "ROUGE_L": rouge_l_score(solution_str, ground_truth),
        "CIDEr": cider_score(solution_str, ground_truth),
        "SPICE": spice_score(solution_str, ground_truth)
    }
    active_metrics = [v for v in metrics.values() if v is not None]
    average = float(sum(active_metrics) / len(active_metrics)) if active_metrics else 0.0
    metrics["score"] = average
    return metrics