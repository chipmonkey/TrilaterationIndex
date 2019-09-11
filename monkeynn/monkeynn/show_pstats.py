import pstats
from pstats import SortKey
p = pstats.Stats('cprofile.out')
# p.strip_dirs().sort_stats(-1).print_stats()
# p.sort_stats(SortKey.NAME)
# p.sort_stats(SortKey.CUMULATIVE).print_stats(10)
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(10)
# p.print_stats()
