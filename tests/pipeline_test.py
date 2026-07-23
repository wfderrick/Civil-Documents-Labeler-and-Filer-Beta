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
    assert vote_for_value((val for val in values), fallback) == fallback

def test_vote_for_value_one_known():
    values = [
            "Unknown",
            "Unknown",
            "test",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
        ]
    fallback = "wrong"
    assert vote_for_value((val for val in values), fallback) == "test"

def test_vote_for_value_tie():
    values = [
            "Unknown",
            "Unknown",
            "test",
            "wrong",
            "test",
            "wrong",
            "Unknown",
            ]
    fallback = "wrong"
    assert vote_for_value((val for val in values), fallback) == "test"

def test_vote_for_value_first_val():
    values = [
            "test",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown"   
        ]
    fallback = "wrong"
    assert vote_for_value((val for val in values), fallback) == "test"

def test_vote_for_value_last_val():
    values = [
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "test",
        ]
    fallback = "wrong"
    assert vote_for_value((val for val in values), fallback) == "test"