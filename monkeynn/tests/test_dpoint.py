from monkeynn.dpoint import dPoint

def testdPoint():
    dp1 = dPoint(5, 2.4)
    dp2 = dPoint(6, 2.7)

    assert dp1 != dp2
    assert dp2 > dp1
    assert dp2 >= dp1
    assert dp1 < dp2
    assert dp1 <= dp2
