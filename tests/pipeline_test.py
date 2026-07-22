from pipeline import vote_for_value


def test_vote_for_value_empty_values():
    values = []
    fallback = "test"
    assert vote_for_value((val.test for val in values), fallback) == fallback


def test_vote_for_value_all_unknown():
    values = [
        "Unknown",
        "Unknown",
        "Unknown",
        "Unknown",
        "Unknown",
        "Unknown",
        "Unknown",
    ]
    fallback = "test"
    assert vote_for_value((val.test for val in values), fallback) == fallback
