import pstats

s = pstats.Stats("QBFProfile_geo.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QKDProfile_geo.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QTRProfile_geo.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QT3Profile_geo.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QT4Profile_geo.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()
