import time
import sys
import argparse
from ChipCnfg import *


def test_config():
    # Note that this should have literally no effect, but should be able to read it.
    zeroes = ['0']*176

    for i in range(176):
        current = zeroes[:]
        current[i] = '1'
        clone = True
        if i == 1: 
            clone = False
        ProgBotCnfgDacs(''.join(current),LD_DACS=False,LD_CTRL=False,clone=clone)  # Write to SR, but don't load it.

    # Write once more to shift out the last pattern.
    ProgBotCnfgDacs(''.join(zeroes),LD_DACS=False,LD_CTRL=False)  # Write to SR, but don't load it.
    return


def test_column(col):
    #first set the global ctrl so that the col recieves the SR clock and input. LD should be 0 at this time
    CtrlBusPatPixCnfg=getCtrlBusPatPixCnfg(col,cnfgbits=LD_IN0_7,lden=LDENABLE_SEL,ss0='0',ss1='0',cnfgmod='00')[::-1]
    ProgBotCnfgDacs(CtrlBusPatPixCnfg,LD_DACS=False,LD_CTRL=True,clone=False) 
    # Now write a pattern to the shift register for that column. 
    # The pattern is taken from which pixels have not all zeroes in the PixMatDict
    rows = [x for x in range(64) if x%2 == 1]
    for row in rows:
        PixMatDict[col][row]='1'*8
    ProgDcolSreg(col)
    return

    
    
def main():
    parser = argparse.ArgumentParser(description="Run a test to write and read a shift register.")
    parser.add_argument('sr', choices=['config','column'], help='Which type of shift register to test.')
    parser.add_argument('-c','--columns', dest='columns', help='Which column(s) to test if column is the selected type. Specify by number(s) or use "all"', default=["all"],nargs='*')

    args = parser.parse_args()
    if args.sr == 'config':
        test_config()
    if args.sr == 'column':
        if 'all' in args.columns:
            cs = range(18)
        else:
            cs = [int(val) for val in args.columns]
        for c in cs:
            test_column(c)
            test_column(c)
    

if __name__ == "__main__":
    main()
