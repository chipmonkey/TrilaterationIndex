# From https://docs.python.org/3/library/profile.html
python -m cProfile -o cprofile.out test_ndim.py
python show_pstats.py
