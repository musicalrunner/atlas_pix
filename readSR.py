import dscope
import argparse

def collect(nsamples, outname, clock, data):
    scope = dscope.ScopeInst(0)
    scope.init_digital_channel(clk=clock)
    traces = [scope.dtrace(timeout=30.0)[0][data] for i in range(nsamples)]
    outfile = open(outname, 'w')
    for trace in traces:
        tmp = [str(b) for b in trace]
        outfile.write(','.join(tmp) + '\n')
    outfile.close()
    return

def main():
    parser = argparse.ArgumentParser(description="Read a shift register (defaults are clock on D0, SR on D1).")
    parser.add_argument('nsamples', type=int, help='The number of samples to read from the shift register.')
    parser.add_argument('outfile', help='Save csv traces to designated file.')
    parser.add_argument('--clock', type=int, help='Specify the channel which samples the clock.',default=0)
    parser.add_argument('--data', type=int, help='Specify the channel which samples the data.',default=1)
    args = parser.parse_args()
    collect(args.nsamples,args.outfile,args.clock,args.data)
    return

if __name__ == "__main__":
    main()
