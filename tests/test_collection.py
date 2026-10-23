"""Tests for collection.py: period parsing, ordering, spreading and writing."""
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import piexif
import pytest
from PIL import Image

from photodate import collection as coll


@pytest.mark.parametrize("name,start,end", [
    ("IX - Okt '88 tot Jan '90", date(1988, 10, 1), date(1990, 1, 31)),
    ("V - Mei '80 tot Okt '83", date(1980, 5, 1), date(1983, 10, 31)),
    ("IV - Juni '78 tot Juni '80", date(1978, 6, 1), date(1980, 6, 30)),
    ("X - Jan '90 tot Jul '90\\", date(1990, 1, 1), date(1990, 7, 31)),
    ("XIV - Jul '94 - Dec ;95", date(1994, 7, 1), date(1995, 12, 31)),
    ("XVII - Juli 99 tot Sep 2000", date(1999, 7, 1), date(2000, 9, 30)),
    ("VII - Aug '85 tot Feb '88", date(1985, 8, 1), date(1988, 2, 29)),
    ("II - Tot Caroline's geboorte 21-9-74", None, date(1974, 9, 21)),
    ("III - Vanaf 21-9-74", date(1974, 9, 21), None),
    ("I - voor het trouwen", None, None),
    ("XII", None, None),
    ("Johanna 50 mapje", None, None),
])
def test_parse_period_real_folder_names(name, start, end):
    assert coll.parse_period(name) == (start, end)


def test_album_order_roman():
    assert coll.album_order("XIV - Jul '94") == 14
    assert coll.album_order("IX - Okt '88") == 9
    assert coll.album_order("XII") == 12
    assert coll.album_order("Johanna 50 mapje") is None
    assert coll.album_order("Italië 1990") is None


def test_order_photos_by_sequence_not_alphabet():
    paths = [Path(n) for n in ["boek_10.jpg", "boek_2.jpg", "boek_1.JPG"]]
    assert [p.name for p in coll.order_photos(paths)] == ["boek_1.JPG", "boek_2.jpg", "boek_10.jpg"]


def test_order_photos_camera_names_are_cover_shots():
    paths = [Path(n) for n in ["vanGilsFotoBoekI_002.jpg", "IMG_2372.jpg", "vanGilsFotoBoekI_001.jpg"]]
    assert coll.order_photos(paths)[0].name == "IMG_2372.jpg"


def test_order_photos_long_unnumbered_run_goes_last():
    paths = [Path(f"boek_{i:03d}.jpg") for i in range(1, 4)] + [Path(f"7616{i}.1.jpg") for i in range(5)]
    ordered = coll.order_photos(paths)
    assert [p.name for p in ordered[:3]] == ["boek_001.jpg", "boek_002.jpg", "boek_003.jpg"]


def test_spread_is_bounded_and_strictly_increasing():
    dts = coll.spread_datetimes(500, date(1988, 10, 1), date(1990, 1, 31))
    assert dts[0].date() == date(1988, 10, 1)
    assert dts[-1].date() == date(1990, 1, 31)
    assert all(a < b for a, b in zip(dts, dts[1:]))


def test_spread_single_photo_and_single_day():
    assert coll.spread_datetimes(1, date(2000, 1, 1), date(2000, 12, 31)) == [datetime(2000, 1, 1, 12, 0)]
    same_day = coll.spread_datetimes(3, date(2000, 1, 1), date(2000, 1, 1))
    assert [d.minute for d in same_day] == [0, 1, 2]


def _make_album(root: Path, name: str, count: int) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    for i in range(1, count + 1):
        Image.new("RGB", (40, 30), (i * 10 % 255, 50, 90)).save(folder / f"boek_{i:03d}.jpg")
    return folder


def test_load_collection_fills_gaps_from_neighbours(tmp_path):
    _make_album(tmp_path, "XI - Jul '90 tot Dec '91", 2)
    _make_album(tmp_path, "XII", 3)
    _make_album(tmp_path, "XIII - Nov '92 tot Jul '94", 2)
    _make_album(tmp_path, "_dubbel", 2)
    albums = coll.load_collection(tmp_path, "Gils")
    assert [a.name for a in albums] == ["XI - Jul '90 tot Dec '91", "XII", "XIII - Nov '92 tot Jul '94"]
    xii = albums[1]
    assert (xii.start, xii.end) == (date(1991, 12, 31), date(1992, 11, 1))
    assert (xii.start_source, xii.end_source) == ("buren", "buren")
    assert xii.rel == "Gils/XII"


def test_manual_period_wins(tmp_path):
    _make_album(tmp_path, "I - voor het trouwen", 2)
    albums = coll.load_collection(tmp_path, "Gils", {"Gils/I - voor het trouwen": ("1968-01", "1972-12")})
    assert (albums[0].start, albums[0].end) == (date(1968, 1, 1), date(1972, 12, 31))
    assert albums[0].complete


def test_apply_and_restore_album(tmp_path):
    _make_album(tmp_path / "Gils", "IX - Okt '88 tot Jan '90", 3)
    album = coll.load_collection(tmp_path / "Gils", "Gils")[0]
    backups = tmp_path / "backups"

    written, errors = coll.apply_album(album, backups)
    assert errors == []
    assert len(written) == 3

    first = album.photos[0]
    exif = piexif.load(str(first))
    assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"1988:10:01 12:00:00"
    description = exif["0th"][piexif.ImageIFD.ImageDescription]
    assert description.startswith(b"Datum geschat: oktober 1988")
    assert b"?" not in description
    assert datetime.fromtimestamp(os.path.getmtime(first)).year == 1988
    assert (backups / first.name).exists()

    assert coll.restore_album(album.folder, backups) == 3
    restored = piexif.load(str(first))
    assert piexif.ExifIFD.DateTimeOriginal not in restored.get("Exif", {})


def test_backup_keeps_first_original(tmp_path):
    _make_album(tmp_path / "Gils", "IX - Okt '88 tot Jan '90", 1)
    album = coll.load_collection(tmp_path / "Gils", "Gils")[0]
    backups = tmp_path / "backups"
    coll.apply_album(album, backups)
    coll.apply_album(album, backups)
    assert piexif.ExifIFD.DateTimeOriginal not in piexif.load(str(backups / "boek_001.jpg")).get("Exif", {})


def test_spread_by_links_same_moment_same_day():
    links = ["nieuw", "moment", "moment", "nieuw", "reeks", "reeks", "nieuw"]
    dts = coll.spread_by_links(links, date(1989, 1, 1), date(1989, 12, 31))
    assert dts[0].date() == dts[1].date() == dts[2].date()
    assert (dts[4].date() - dts[3].date()).days == 1
    assert (dts[5].date() - dts[4].date()).days == 1
    assert dts[-1].date() == date(1989, 12, 31)
    assert all(a < b for a, b in zip(dts, dts[1:]))


def test_spread_by_links_without_new_events_spreads_series():
    dts = coll.spread_by_links(["nieuw", "reeks", "reeks"], date(2000, 1, 1), date(2000, 1, 11))
    assert [d.day for d in dts] == [1, 6, 11]


def test_spread_by_links_short_period_stays_in_bounds():
    links = ["nieuw"] + ["reeks"] * 30 + ["nieuw"] * 5
    dts = coll.spread_by_links(links, date(2000, 1, 1), date(2000, 1, 10))
    assert dts[-1].date() <= date(2000, 1, 10)
    assert all(a < b for a, b in zip(dts, dts[1:]))


def test_moment_count():
    assert coll.moment_count(["nieuw", "moment", "reeks", "moment", "nieuw"]) == 3
    assert coll.moment_count([]) == 0


def test_planned_uses_links_when_complete(tmp_path):
    _make_album(tmp_path / "Gils", "IX - Okt '88 tot Jan '90", 3)
    album = coll.load_collection(tmp_path / "Gils", "Gils")[0]
    album.links = ["nieuw", "moment", "moment"]
    days = {dt.date() for _, dt in album.planned()}
    assert days == {date(1988, 10, 1)}
    album.links = ["nieuw", "moment"]  # stale: wrong length -> ignored
    assert len({dt.date() for _, dt in album.planned()}) == 3


def _days(dts):
    return [d.date() for d in dts]


def test_seasons_pull_summer_photos_before_an_autumn_start():
    # Album "Okt '88 tot Jan '90": the first moments are clearly summer.
    links = ["nieuw", "moment", "nieuw", "moment"] + ["nieuw"] * 20
    seasons = ["zomer", "zomer", "zomer", "zomer"] + ["onbekend"] * 20
    dts = coll.place_with_seasons(links, seasons, date(1988, 10, 1), date(1990, 1, 31))
    days = _days(dts)
    assert days[0].month in (6, 7, 8, 9) and days[0].year == 1988
    assert days[2].month in (6, 7, 8, 9)
    assert days[0] >= date(1988, 10, 1) - timedelta(days=coll.SLACK_DAYS)
    assert all(a < b for a, b in zip(dts, dts[1:]))


def test_seasons_put_winter_moment_in_winter():
    links = ["nieuw"] * 30
    seasons = ["onbekend"] * 30
    seasons[10] = "winter"  # evenly this would land in early summer
    dts = coll.place_with_seasons(links, seasons, date(1989, 1, 1), date(1989, 12, 31))
    assert dts[10].month in (11, 12, 1, 2, 3)
    assert all(a < b for a, b in zip(dts, dts[1:]))


def test_without_seasons_placement_matches_even_spread():
    links = ["nieuw", "moment", "reeks", "nieuw", "nieuw", "moment", "nieuw"]
    seasons = ["onbekend"] * len(links)
    placed = _days(coll.place_with_seasons(links, seasons, date(1990, 1, 1), date(1990, 7, 31)))
    even = _days(coll.spread_by_links(links, date(1990, 1, 1), date(1990, 7, 31)))
    assert placed == even


def test_series_stay_on_consecutive_days_with_seasons():
    links = ["nieuw", "reeks", "reeks", "reeks", "nieuw", "nieuw"]
    seasons = ["zomer"] * 4 + ["onbekend"] * 2
    days = _days(coll.place_with_seasons(links, seasons, date(1989, 1, 1), date(1989, 12, 31)))
    assert [(b - a).days for a, b in zip(days[:4], days[1:4])] == [1, 1, 1]
    assert days[0].month in (6, 7, 8)


def test_planned_prefers_seasons_when_present(tmp_path):
    _make_album(tmp_path / "Gils", "IX - Okt '88 tot Jan '90", 3)
    album = coll.load_collection(tmp_path / "Gils", "Gils")[0]
    album.links = ["nieuw", "nieuw", "nieuw"]
    album.seasons = ["zomer", "onbekend", "onbekend"]
    first = album.planned()[0][1]
    assert first.date() < date(1988, 10, 1)


GILS = [
    coll.Person("Gerard", date(1945, 11, 7), date(1999, 4, 6)),
    coll.Person("Johanna", date(1951, 8, 13)),
    coll.Person("Caroline", date(1974, 9, 21)),
    coll.Person("Marijke", date(1976, 5, 10)),
    coll.Person("Gerd-Jan", date(1980, 1, 25)),
]


def test_easter_sunday():
    assert coll.easter_sunday(1989) == date(1989, 3, 26)
    assert coll.easter_sunday(2000) == date(2000, 4, 23)


def test_holiday_windows():
    first, last = date(1988, 6, 1), date(1990, 3, 31)
    assert (date(1988, 11, 15), date(1988, 12, 6)) in coll.holiday_windows("sinterklaas", first, last)
    assert coll.holiday_windows("koninginnedag", first, last) == [(date(1989, 4, 30), date(1989, 4, 30))]
    nine = coll.holiday_windows("verjaardag", first, last, GILS, age=9)
    assert nine == [(date(1989, 1, 22), date(1989, 2, 1))]  # only Gerd-Jan turns 9 in this period
    any_birthday = coll.holiday_windows("verjaardag", first, last, GILS)
    assert len(any_birthday) >= 9


def test_no_birthdays_after_death():
    windows = coll.holiday_windows("verjaardag", date(1999, 6, 1), date(2000, 9, 30), GILS[:1])
    assert windows == []


def test_sinterklaas_moment_lands_in_its_window():
    links = ["nieuw"] * 40
    holidays = ["geen"] * 40
    holidays[5] = "sinterklaas"  # evenly this would be around January 1989
    dts = coll.place_with_seasons(links, ["onbekend"] * 40, date(1988, 10, 1), date(1990, 1, 31),
                                  holidays=holidays)
    assert date(1988, 11, 15) <= dts[5].date() <= date(1988, 12, 6)
    assert all(a < b for a, b in zip(dts, dts[1:]))


def test_birthday_with_visible_age_uses_family():
    links = ["nieuw"] * 40
    holidays = ["geen"] * 40
    ages = [None] * 40
    holidays[12], ages[12] = "verjaardag", 9  # evenly around the end of February 1989
    dts = coll.place_with_seasons(links, ["onbekend"] * 40, date(1988, 10, 1), date(1990, 1, 31),
                                  holidays=holidays, ages=ages, family=GILS)
    assert date(1989, 1, 22) <= dts[12].date() <= date(1989, 2, 1)


def test_description_mentions_holiday(tmp_path):
    _make_album(tmp_path / "Gils", "IX - Okt '88 tot Jan '90", 3)
    album = coll.load_collection(tmp_path / "Gils", "Gils")[0]
    album.links = ["nieuw", "moment", "nieuw"]
    album.holidays = ["sinterklaas", "sinterklaas", "geen"]
    album.seasons = ["onbekend"] * 3
    coll.apply_album(album, tmp_path / "backups")
    desc = piexif.load(str(album.photos[0]))["0th"][piexif.ImageIFD.ImageDescription]
    assert b", Sinterklaas (album IX" in desc
    assert piexif.load(str(album.photos[0]))["Exif"][piexif.ExifIFD.DateTimeOriginal][:7] in (b"1988:11", b"1988:12")


def test_carnaval_driekoningen_sintmaarten_windows():
    first, last = date(1988, 10, 1), date(1990, 1, 31)
    assert (date(1989, 2, 4), date(1989, 2, 7)) in coll.holiday_windows("carnaval", first, last)
    assert (date(1989, 1, 5), date(1989, 1, 7)) in coll.holiday_windows("driekoningen", first, last)
    assert (date(1988, 11, 10), date(1988, 11, 12)) in coll.holiday_windows("sintmaarten", first, last)


def test_misrecognised_holiday_does_not_drag_a_run_of_moments():
    # Album IX: Sinterklaas, kerst (2 moments), Driekoningen, then carnaval, of which one moment
    # was wrongly called Sinterklaas. Kerst must stay on 24-26 December.
    labels = (["geen"] * 6 + ["sinterklaas"] + ["geen"] * 2 + ["kerst", "kerst", "driekoningen"]
              + ["carnaval"] * 3 + ["sinterklaas"] + ["carnaval"] * 2 + ["geen"] * 22)
    links = ["nieuw"] * len(labels)
    dts = coll.place_with_seasons(links, ["onbekend"] * len(labels), date(1988, 10, 1),
                                  date(1990, 1, 31), holidays=labels)
    days = [d.date() for d in dts]
    assert date(1988, 11, 15) <= days[6] <= date(1988, 12, 6)
    assert date(1988, 12, 24) <= days[9] < days[10] <= date(1988, 12, 26)
    assert date(1989, 1, 5) <= days[11] <= date(1989, 1, 7)
    assert all(date(1989, 1, 20) <= d <= date(1989, 2, 20) for d in days[12:18])
