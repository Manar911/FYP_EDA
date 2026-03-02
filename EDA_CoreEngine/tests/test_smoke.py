from eda.airport_db import load_airports


def test_smoke():
    assert 1 + 1 == 2


def test_airports_load():
    airports = load_airports()
    assert len(airports) >= 1
    assert airports[0].icao