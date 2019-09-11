# From https://docs.python.org/3/library/profile.html
python -m cProfile -o cprofile.out testndimm.py
python show_pstats.py
