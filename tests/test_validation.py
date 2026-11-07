from gmdh_conformal.validation import validate_results


def test_archived_followup_bundle_is_complete_and_matches_reported_gmdh_values():
    result = validate_results()
    assert result == {
        "fold_files": 25,
        "pnf_files": 25,
        "rows": 1250,
        "reproduction_checks": 5,
    }
