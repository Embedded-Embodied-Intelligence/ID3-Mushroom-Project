"""Figures are produced with non-empty files; the tree renders via Graphviz or the matplotlib fallback."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import src.visualization as viz  # noqa: E402
from src.evaluation import confusion_matrix_frame  # noqa: E402
from src.id3 import ID3Classifier  # noqa: E402
from src.preprocessing import CategoricalPreprocessor, feature_profile  # noqa: E402


@pytest.fixture(scope="module")
def mushroom_tree(mushroom_split):
    X, y, split = mushroom_split
    pre = CategoricalPreprocessor("missing_category").fit(X.loc[split.train])
    return ID3Classifier(class_priority=("p", "e")).fit(pre.transform(X.loc[split.train]), y.loc[split.train])


def _saved(fig, tmp_path, name, formats=("png",)):
    paths = viz.save_figure(fig, tmp_path / name, formats)
    plt.close(fig)
    return paths


def test_chapter_figures_have_non_empty_files(tmp_path, mushroom_df, mushroom_split):
    viz.set_presentation_style()
    X, y, _ = mushroom_split
    figures = {
        "entropy": viz.plot_entropy_curve(("Mushroom", 0.482)),
        "overview": viz.plot_dataset_overview(y, feature_profile(X)),
        "missing": viz.plot_missing_overview(mushroom_df),
        "cm": viz.plot_confusion_matrix(confusion_matrix_frame(["e", "p", "p"], ["e", "p", "e"]), "test"),
    }
    for name, fig in figures.items():
        for path in _saved(fig, tmp_path, name, ("png", "svg")):
            assert path.stat().st_size > 5_000, path


@pytest.mark.parametrize("options", [{}, {"group_pure_leaves": True}, {"orientation": "LR"},
                                     {"subtree_path": [("odor", "n")]}, {"highlight_node_ids": [0]}])
def test_tree_diagram_renders(tmp_path, mushroom_tree, options):
    paths = _saved(viz.plot_tree(mushroom_tree, title="drzewo", **options), tmp_path, "tree", ("png", "svg"))
    assert all(p.stat().st_size > 20_000 for p in paths)


def test_dot_export_describes_every_node_and_branch(mushroom_tree):
    dot = viz.tree_to_dot(mushroom_tree)
    assert dot.startswith("digraph ID3 {") and dot.rstrip().endswith("}")
    assert dot.count(" -> ") == mushroom_tree.node_count_ - 1
    assert sum(line.strip().startswith("n") and "[label=<" in line and "->" not in line
               for line in dot.splitlines()) == mushroom_tree.node_count_
    assert "odor" in dot and "none (n)" in dot


def test_render_falls_back_to_matplotlib_without_dot(tmp_path, monkeypatch, mushroom_tree):
    monkeypatch.setattr(viz, "graphviz_available", lambda: False)
    result = viz.render_tree(mushroom_tree, tmp_path / "final_tree")
    assert result["backend"] == "matplotlib"
    assert result["dot"].read_text().startswith("digraph")
    assert {p.suffix for p in result["files"]} == {".svg", ".png"}
    assert all(p.stat().st_size > 20_000 for p in result["files"])


@pytest.mark.skipif(not viz.graphviz_available(), reason="Graphviz 'dot' is not installed on this machine")
def test_render_with_graphviz_when_available(tmp_path, mushroom_tree):
    result = viz.render_tree(mushroom_tree, tmp_path / "final_tree")
    assert result["backend"] == "graphviz" and all(p.stat().st_size > 1_000 for p in result["files"])


def _luminance(hex_color):
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def test_class_colors_are_distinct_and_contrast_with_the_surface():
    surface = _luminance(viz.SURFACE)
    for color in viz.CLASS_COLORS.values():
        ratio = (surface + 0.05) / (_luminance(color) + 0.05)
        assert ratio >= 3.0, (color, ratio)
    assert viz.CLASS_COLORS["e"] != viz.CLASS_COLORS["p"]


def test_graphviz_code_path_with_a_stub_dot_program(tmp_path, monkeypatch, mushroom_tree):
    """Exercises the Graphviz branch (subprocess call, formats, file names) without a real installation."""
    import os
    import stat

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "dot"
    stub.write_text('#!/bin/sh\nout=""; src=""\nwhile [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift 2;; '
                    '-T*) shift;; *) src="$1"; shift;; esac; done\n'
                    'head -c 7 "$src" > "$out"; printf " stub-render" >> "$out"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    result = viz.render_tree(mushroom_tree, tmp_path / "final_tree")
    assert result["backend"] == "graphviz"
    assert [p.suffix for p in result["files"]] == [".svg", ".png"]
    assert all(p.read_text().startswith("digraph stub-render") for p in result["files"])
