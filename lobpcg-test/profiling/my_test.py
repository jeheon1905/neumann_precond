from cProfile import Profile
from pstats import Stats
import sys

file = sys.argv[1]
stats = Stats(file)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats() 


