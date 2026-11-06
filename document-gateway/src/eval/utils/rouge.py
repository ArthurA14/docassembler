from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

from rouge_score import rouge_scorer, scoring
from tqdm import tqdm


class Rouge:
    """Computes ROUGE scores for evaluating text assembly quality."""
    def _compute(
        self,
        predictions: List[str],
        references: List[List[str]],
        rouge_types: List[str] = ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_aggregator: bool = True,
        use_stemmer: bool = True
    ) -> Dict[str, Any]:
        """
        Compute ROUGE scores for predictions against multiple references.

        Args:
            predictions: List of predicted texts.
            references: List of lists of reference texts (multiple refs per prediction).
            rouge_types: ROUGE metrics to compute (default: ["rouge1", "rouge2", "rougeL", "rougeLsum"]).
            use_aggregator: Whether to aggregate scores across examples (default: True).
            use_stemmer: Whether to use stemming for tokenization (default: True).

        Returns:
            Dict containing ROUGE scores, either aggregated or per-example.
        """
        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
        aggregator = None
        scores: List[Dict[str, Any]] = []
        result: Dict[str, Any] = {}
        if use_aggregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores = []

        for ref, pred in tqdm(zip(references, predictions), total=len(references), desc="Calculating ROUGE"):
            ex_scores: List[Dict[str, Any]] = []
            for r in ref:
                score: dict[str, float] = scorer.score(r, pred)  # returns a dict of scores
                ex_scores.append(score)

            # need to aggregate across references
            # we do so by taking the score with the highest f-score
            # (keeping the precision and recall even if it's lower than p/r wrt another ref)
            # and do this for all references
            max_refs: DefaultDict[str, int] = defaultdict(int)
            max_fmeasures: DefaultDict[str, float] = defaultdict(float)
            for i, s in enumerate(ex_scores):
                for rouge_type, rouge_score in s.items():
                    if rouge_score.fmeasure > max_fmeasures[rouge_type]:
                        max_fmeasures[rouge_type] = rouge_score.fmeasure
                        max_refs[rouge_type] = i

            ex_score: Dict[str, Any] = {}
            for rouge_type, max_ref in max_fmeasures.items():
                ex_score[rouge_type] = ex_scores[max_refs[rouge_type]][rouge_type]

            if aggregator:
                aggregator.add_scores(ex_score)
            else:
                scores.append(ex_score)

        if aggregator:
            result = aggregator.aggregate()
        else:
            result = {}
            for key in scores[0]:
                result[key] = list(score[key] for score in scores)

        if isinstance(result, Exception):
            raise result

        return result
