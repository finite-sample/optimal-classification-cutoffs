"""Vulture whitelist for false positives (public API functions)."""

# bayes_core.py - methods called via class instances
from_costs  # noqa (UtilitySpec.from_costs classmethod)
decide  # noqa (BayesOptimal.decide method)

# core.py - enum value
WEIGHTED  # noqa (Average enum value)

# metrics/__init__.py - public API
register  # noqa
list_available  # noqa
info  # noqa

# optimize.py - legacy API tested directly
find_optimal_threshold  # noqa
find_optimal_threshold_multiclass  # noqa

# validation.py - internal utilities tested directly
validate_inputs  # noqa
_validate_threshold  # noqa
_validate_threshold_inputs  # noqa
get_sample_weights  # noqa

# metrics_core.py - used in tests and internal code
multiclass_confusion_matrices_at_thresholds  # noqa
multiclass_metric_single_label  # noqa
register_metrics  # noqa
needs_probability_scores  # noqa
has_vectorized_implementation  # noqa
ovr_confusion_counts  # noqa
compute_multiclass_metrics_from_labels  # noqa
make_cost_metric  # noqa

# validation.py - validators used internally
validate_classification  # noqa
_validate_metric_name  # noqa
_validate_averaging_method  # noqa
_validate_optimization_method  # noqa

# algorithms/expected.py - public API wrapper
dinkelbach_fbeta_multilabel  # noqa
