#!/usr/bin/env python3
"""Tests for the pure parts, where a mistake stays silent.

Run with `python -m unittest test_group_scenes` or `python
test_group_scenes.py`. Nothing here reads a photo or calls exiftool.
"""

import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

import group_scenes as gs

SUBJECT = gs.SUBJECT
HIERARCHICAL = gs.HIERARCHICAL

START = datetime(2020, 5, 23, 13, 45, 9).timestamp()
NAME = "Scene_20200523_134509"


def matrix(size, pairs, high=0.9, low=0.1):
    """Similarities where the listed pairs match and nothing else does."""
    sim = [[high if i == j else low for j in range(size)] for i in range(size)]
    for a, b in pairs:
        sim[a][b] = sim[b][a] = high
    return sim


class SceneNames(unittest.TestCase):
    def test_name_comes_from_the_earliest_member(self):
        self.assertEqual(gs.scene_names([[0, 1]], [START, START + 11]), [NAME])

    def test_order_inside_the_group_does_not_matter(self):
        self.assertEqual(gs.scene_names([[1, 0]], [START + 11, START]), [NAME])

    def test_name_survives_the_album_growing(self):
        # a photo arrives before the group, so every index shifts
        times = [START - 500, START, START + 11]
        self.assertEqual(gs.scene_names([[1, 2]], times), [NAME])

    def test_two_scenes_in_the_same_second_get_a_suffix(self):
        times = [START, START + 1, START, START + 1]
        self.assertEqual(
            gs.scene_names([[0, 1], [2, 3]], times), [NAME, NAME + "_2"]
        )

    def test_no_scenes_gives_no_names(self):
        self.assertEqual(gs.scene_names([], []), [])


class TagOps(unittest.TestCase):
    def test_clean_file_gets_both_tags(self):
        self.assertEqual(
            gs.tag_ops([], [], NAME, "Scenes"),
            (f"-{SUBJECT}+={NAME}", f"-{HIERARCHICAL}+=Scenes|{NAME}"),
        )

    def test_tagged_file_is_left_alone(self):
        self.assertEqual(
            gs.tag_ops([NAME], [f"Scenes|{NAME}"], NAME, "Scenes"), ()
        )

    def test_doubled_value_collapses_to_one(self):
        self.assertEqual(
            gs.tag_ops([NAME, NAME], [f"Scenes|{NAME}"] * 2, NAME, "Scenes"),
            (
                f"-{SUBJECT}-={NAME}",
                f"-{SUBJECT}+={NAME}",
                f"-{HIERARCHICAL}-=Scenes|{NAME}",
                f"-{HIERARCHICAL}+=Scenes|{NAME}",
            ),
        )

    def test_number_from_the_old_scheme_is_replaced(self):
        self.assertEqual(
            gs.tag_ops(["Scene_01"], ["Scenes|Scene_01"], NAME, "Scenes"),
            (
                f"-{SUBJECT}-=Scene_01",
                f"-{SUBJECT}+={NAME}",
                f"-{HIERARCHICAL}-=Scenes|Scene_01",
                f"-{HIERARCHICAL}+=Scenes|{NAME}",
            ),
        )

    def test_three_digit_number_is_ours_too(self):
        self.assertEqual(
            gs.tag_ops(["Scene_104"], [], NAME, "Scenes"),
            (
                f"-{SUBJECT}-=Scene_104",
                f"-{SUBJECT}+={NAME}",
                f"-{HIERARCHICAL}+=Scenes|{NAME}",
            ),
        )

    def test_foreign_keywords_are_never_touched(self):
        self.assertEqual(
            gs.tag_ops(
                ["Flowers", "Minsk", NAME],
                ["Trips|Minsk|Garden", f"Scenes|{NAME}"],
                NAME,
                "Scenes",
            ),
            (),
        )

    def test_a_lookalike_keyword_is_not_ours(self):
        self.assertEqual(
            gs.tag_ops(
                ["Scenery", "Scene_of_crime"], ["Scenes|Holiday"],
                NAME, "Scenes",
            ),
            (f"-{SUBJECT}+={NAME}", f"-{HIERARCHICAL}+=Scenes|{NAME}"),
        )

    def test_ungrouped_file_loses_only_our_tags(self):
        self.assertEqual(
            gs.tag_ops(
                ["Flowers", "Scene_01"], ["Trips|Minsk", "Scenes|Scene_01"],
                None, "Scenes",
            ),
            (f"-{SUBJECT}-=Scene_01", f"-{HIERARCHICAL}-=Scenes|Scene_01"),
        )

    def test_another_parent_leaves_the_scenes_tree_alone(self):
        self.assertEqual(
            gs.tag_ops([], ["Scenes|Scene_01"], NAME, "Groups"),
            (f"-{SUBJECT}+={NAME}", f"-{HIERARCHICAL}+=Groups|{NAME}"),
        )

    def test_only_the_hierarchy_half_is_stale(self):
        self.assertEqual(
            gs.tag_ops([NAME], ["Scenes|Scene_01"], NAME, "Scenes"),
            (
                f"-{HIERARCHICAL}-=Scenes|Scene_01",
                f"-{HIERARCHICAL}+=Scenes|{NAME}",
            ),
        )


class Cluster(unittest.TestCase):
    def cluster(self, times, pairs, threshold=0.75, gap=180.0, **kwargs):
        order = sorted(range(len(times)), key=lambda i: times[i])
        sim = matrix(len(times), pairs, **kwargs)
        return gs.cluster(order, sim, times, threshold, gap)

    def test_similar_and_close_in_time_group(self):
        self.assertEqual(self.cluster([0, 5], [(0, 1)]), [[0, 1]])

    def test_similar_but_past_the_gap_do_not_group(self):
        self.assertEqual(self.cluster([0, 500], [(0, 1)]), [])

    def test_close_in_time_but_different_do_not_group(self):
        self.assertEqual(self.cluster([0, 5], []), [])

    def test_threshold_is_inclusive(self):
        self.assertEqual(self.cluster([0, 5], [(0, 1)], high=0.75), [[0, 1]])

    def test_just_under_the_threshold_does_not_group(self):
        self.assertEqual(self.cluster([0, 5], [(0, 1)], high=0.749), [])

    def test_a_chain_is_refused(self):
        # A matches B and B matches C, but A and C have nothing in common
        self.assertEqual(self.cluster([0, 5, 10], [(0, 1), (1, 2)]), [[0, 1]])

    def test_mutually_similar_shots_form_one_group(self):
        pairs = [(0, 1), (1, 2), (0, 2)]
        self.assertEqual(self.cluster([0, 5, 10], pairs), [[0, 1, 2]])

    def test_lone_photos_are_dropped(self):
        self.assertEqual(self.cluster([0, 5, 400], [(0, 1)]), [[0, 1]])

    def test_a_rejected_photo_starts_the_next_group(self):
        pairs = [(0, 1), (2, 3)]
        self.assertEqual(self.cluster([0, 5, 10, 15], pairs), [[0, 1], [2, 3]])

    def test_the_gap_is_measured_from_the_previous_member(self):
        # each step is inside the gap, so the group spans more than the gap
        pairs = [(0, 1), (1, 2), (0, 2)]
        self.assertEqual(self.cluster([0, 100, 200], pairs), [[0, 1, 2]])

    def test_members_come_back_in_time_order(self):
        self.assertEqual(self.cluster([10, 0], [(0, 1)]), [[1, 0]])

    def test_nothing_in_nothing_out(self):
        self.assertEqual(self.cluster([], []), [])


class LazySimilarity(unittest.TestCase):
    def setUp(self):
        raw = np.random.default_rng(7).standard_normal((12, 8))
        self.vectors = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        self.matrix = (self.vectors @ self.vectors.T).tolist()
        self.lazy = gs.Similarity(self.vectors)

    def test_every_cell_matches_the_full_matrix(self):
        for i in range(len(self.vectors)):
            for j in range(len(self.vectors)):
                self.assertAlmostEqual(self.lazy[i][j], self.matrix[i][j])

    def test_a_photo_with_itself_is_one(self):
        for i in range(len(self.vectors)):
            self.assertAlmostEqual(self.lazy[i][i], 1.0)

    def test_reading_is_symmetric(self):
        self.assertAlmostEqual(self.lazy[3][7], self.lazy[7][3])

    def test_a_cell_is_a_plain_float(self):
        # the report formats these, and cluster compares them
        self.assertIsInstance(self.lazy[0][1], float)

    def test_clustering_agrees_with_the_full_matrix(self):
        times = [float(i * 10) for i in range(len(self.vectors))]
        order = sorted(range(len(times)), key=lambda i: times[i])
        for threshold in (-0.5, 0.0, 0.3):
            self.assertEqual(
                gs.cluster(order, self.matrix, times, threshold, 180.0),
                gs.cluster(order, self.lazy, times, threshold, 180.0),
            )


class Chunks(unittest.TestCase):
    def setUp(self):
        self.many = [Path("/x" * 50 + str(i)) for i in range(5000)]

    def test_every_path_appears_once(self):
        batches = list(gs.chunks(self.many))
        self.assertEqual([p for b in batches for p in b], self.many)

    def test_no_batch_exceeds_the_budget(self):
        for batch in gs.chunks(self.many):
            room = sum(len(str(p)) + 1 for p in batch)
            self.assertLessEqual(room, gs.ARG_BUDGET)

    def test_empty_input(self):
        self.assertEqual(list(gs.chunks([])), [])

    def test_one_path_longer_than_the_budget_still_comes_back(self):
        huge = [Path("/" + "y" * (gs.ARG_BUDGET * 2))]
        self.assertEqual(list(gs.chunks(huge)), [huge])


class AsList(unittest.TestCase):
    def test_missing_tag(self):
        self.assertEqual(gs.as_list(None), [])

    def test_single_value_becomes_a_list(self):
        self.assertEqual(gs.as_list("Scene_01"), ["Scene_01"])

    def test_list_stays_a_list(self):
        self.assertEqual(gs.as_list(["a", "b"]), ["a", "b"])

    def test_numeric_keyword_becomes_text(self):
        self.assertEqual(gs.as_list(2020), ["2020"])


class Plural(unittest.TestCase):
    def test_one(self):
        self.assertEqual(gs.plural(1, "file"), "1 file")

    def test_many(self):
        self.assertEqual(gs.plural(7, "file"), "7 files")

    def test_none(self):
        self.assertEqual(gs.plural(0, "file"), "0 files")


if __name__ == "__main__":
    unittest.main(verbosity=2)
