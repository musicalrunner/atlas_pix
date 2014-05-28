import pstats
import sys

def main(args):
    for arg in args:
        p = pstats.Stats(arg)
        p.strip_dirs().sort_stats('cumulative').print_stats(30)
        print "\n"


if __name__=="__main__":
    main(sys.argv[1:])
