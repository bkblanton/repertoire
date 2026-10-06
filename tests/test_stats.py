import numpy as np
import pytest

from repertoire_score.stats import cell, connected_groups, correlation, rank


def test_overlap_groups_include_indirect_links():
    assert connected_groups([{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'e'}]) == [[0, 1, 2], [3]]


def test_midpoint_ranks_and_weighted_correlation():
    assert rank(np.array([3, 1, 1, 4])) == pytest.approx([3, 1.5, 1.5, 4])
    x, y, w = np.array([1, 2, 5]), np.array([4, 3, 8]), np.array([1, 2, 4])
    assert correlation(x, y, w) == pytest.approx(np.corrcoef(np.repeat(x, w), np.repeat(y, w))[0, 1])
    assert np.isnan(correlation([1], [2]))
    assert np.isnan(correlation([1, 1, 1], [1, 2, 3]))


def test_cell_formats_estimates_and_intervals():
    result = {'pearson': .5, 'slope_pp_per_move': 1.25, 'spearman': None,
              'confidence_intervals_95': {'pearson': {'bounds': [.1, .9]}, 'slope_pp_per_move': {'bounds': None}}}
    assert cell(result, 'pearson') == '0.500 [0.100, 0.900]'
    assert cell(result, 'slope_pp_per_move') == '1.250% (interval unavailable)'
    assert cell(result, 'spearman') == 'undefined'
