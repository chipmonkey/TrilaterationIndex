from monkeynn.toplist import toplist


def test_pushpop():
    x = toplist(5)
    assert x.numItems == 5
    assert x.count == 0
    assert x.maxP() is None

    x.push(6)
    assert x.dPList == [6]
    assert x.count == 1

    x.push(7)
    assert x.dPList == [6, 7]
    assert x.count == 2

    x.push(5)
    assert x.dPList == [5, 6, 7]
    assert x.count == 3

    x.push(5)
    assert x.dPList == [5, 5, 6, 7]
    assert x.count == 4

    x.push(7)
    assert x.dPList == [5, 5, 6, 7, 7]
    assert x.count == 5

    x.push(11)
    assert x.dPList == [5, 5, 6, 7, 7]
    assert x.count == 5

    x.push(4)
    assert x.dPList == [4, 5, 5, 6, 7]
    assert x.count == 5

    x.push(6)
    assert x.dPList == [4, 5, 5, 6, 6]
    assert x.count == 5

    x.pop(6)
    assert x.dPList == [4, 5, 5, 6]
    assert x.count == 4

    x.pop(4)
    assert x.dPList == [5, 5, 6]
    assert x.count == 3

    x.push(11)
    assert x.dPList == [5, 5, 6, 11]
    assert x.count == 4
    assert x.maxP() == 11

    x.pop(11)
    assert x.dPList == [5, 5, 6]
    assert x.count == 3

    x.poptop()
    assert x.dPList == [5, 5]
    assert x.count == 2
    assert x.maxP() == 5
