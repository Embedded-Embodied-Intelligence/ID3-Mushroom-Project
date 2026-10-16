"""Delivery archives: exclusions, size limit and separation of the answer key."""
import sys
import zipfile

import pytest

from tests.conftest import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import package_submission as pkg  # noqa: E402


@pytest.fixture(scope="module")
def archives(tmp_path_factory):
    return pkg.build_all(out_dir=tmp_path_factory.mktemp("submission"), require_executed=False)


def test_exclusion_rules():
    from pathlib import Path
    for path in (".venv/lib/x.py", "src/__pycache__/id3.cpython-312.pyc", ".pytest_cache/v", ".git/HEAD",
                 "notebooks/.ipynb_checkpoints/a.ipynb", "data/mushroom.zip.part", ".DS_Store",
                 "group_exercise/instructor_private/ANSWER_KEY.md"):
        assert pkg.is_excluded(Path(path)), path
    assert not pkg.is_excluded(Path("src/id3.py"))


def test_main_archive_is_complete_clean_and_small(archives):
    audit = pkg.audit_zip(archives["main"])
    assert audit["corrupt"] is None and not audit["forbidden"] and not audit["private"]
    assert audit["size"] < pkg.MAX_BYTES
    for required in ("data/mushroom.zip", "notebooks/ID3_Mushroom_Project.ipynb", "src/id3.py", "requirements.txt",
                     "group_exercise/ID3_Delivery_Workshop.ipynb"):
        assert any(n.endswith(required) for n in audit["names"]), required
    assert all(n.startswith(pkg.ARCHIVE_ROOT + "/") for n in audit["names"])


def test_student_archive_has_no_answer_key(archives):
    with zipfile.ZipFile(archives["students"]) as archive:
        names = archive.namelist()
    assert sorted(n.split("/", 1)[1] for n in names) == ["ID3_Delivery_Workshop.ipynb", "README.md", "delivery_data.csv"]
    assert not any("instructor" in n or "ANSWER" in n for n in names)


def test_instructor_archive_is_separate(archives):
    with zipfile.ZipFile(archives["instructor"]) as archive:
        assert sorted(n.split("/", 1)[1] for n in archive.namelist()) == ["ANSWER_KEY.md", "ID3_Delivery_Solution.ipynb"]
