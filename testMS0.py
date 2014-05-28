import visa
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack

THRESHOLD = 0.75


def binary_list(x):
    return binary_list(x/2) + [x%2] if x > 1 else [x]


def filled_binary_list(x,nbits=32):
    vals = binary_list(x)
    return [0]*(nbits - len(vals)) + vals


class ScopeInst(visa.Instrument):
    def __init__(self,USB_num=0,GetInstId=0):
        self.InstId = ""
        self.dwidth = 1
        self.clock = 0
        available = [ inst for inst in visa.get_instruments_list() if ("USB%i" % USB_num) in inst]
        if len(available) == 0:
            print "No USB%i instrument found." % i
            sys.exit(1)
        visa.Instrument.__init__(self,available[0])
        if GetInstId == 1:
            self.write("*IDN?")
            self.InstId=self.read()

    def sendStr(self,strIn):
        self.write(strIn)
        return

    def readStr(self):
        return self.read()

    def init_analog_channel(self,channel,data_width=1):
        self.dwidth = data_width
        self.write('DATA:SOU CH1')
        self.write('DATA:WIDTH %i' % self.dwidth)   
        self.write('DATA:ENC FAS')

    def atrace(self,timeout=30.0):
        ymult = float(self.ask('WFMPRE:YMULT?'))
        yzero = float(self.ask('WFMPRE:YZERO?'))
        yoff  = float(self.ask('WFMPRE:YOFF?'))
        xincr = float(self.ask('WFMPRE:XINCR?'))
        # Ask for single trigger and wait for it, to ensure a single waveform.
        self.write('ACQ:STOPA SEQ')
        startt = time.time()
        while '1' in scope.ask("ACQ:STATE?"):
            time.sleep(0.1)
            if time.time() > (startt + timeout):
                self.write('TRIGGER FORCE')
                time.sleep(0.5)
                break
        # Binary format, which should be fastest:
        self.write('CURVE?')
        data = self.read_raw()
        headerlen = 2 + int(data[1])
        header = data[:headerlen]
        vals = data[headerlen:-1]
        if self.dwidth == 1:
            vals = np.array(unpack('>%sb' % len(vals),vals))
        elif self.dwidth == 2:
            vals = np.array(unpack('>%sh' % (len(vals)/2),vals))
        volts = (vals - yoff) * ymult  + yzero
        times = np.arange(len(volts))* xincr
        return times,volts

    def init_digital_channel(self,clk=0):
        self.clock = clk
        self.write('DATA:SOU DIG')
        self.write('DATA:WIDTH 4')
        self.write('DATA:ENC FAS')
        self.write('TRIG:A:LOGI:INP:CLOC:SOU D%i' % self.clock)
        self.write('TRIG:A:LOGI:INP:CLOCK:EDGE RIS')
        for i in range(16):
            self.write('D%i:THR %.2f' % (i,THRESHOLD))
        
    def dtrace(self,timeout=30.0):
        # Ask for single trigger and wait for it, to ensure a single waveform.
        self.write('ACQ:STOPA SEQ')
        self.write("ACQ:STATE RUN")
        startt = time.time()
        while '1' in scope.ask("ACQ:STATE?"):
            time.sleep(0.1)
            if time.time() > (startt + timeout):
                self.write('TRIGGER FORCE')
                time.sleep(0.5)
                break
        # Binary format, which should be fastest:
        self.write('CURVE?')
        data = self.read_raw()
        headerlen = 2 + int(data[1])
        header = data[:headerlen]
        vals = data[headerlen:-1]
        vals = np.array(unpack('>%si' % (len(vals)/4),vals))
        d_lists = [filled_binary_list(val) for val in vals]
        D = np.array([[x[i] for x in d_lists] for i in range(32) ])
        D = D[::-1]
        D = D[:16]
        # Now interpret each channel based on the clock.
        # We want to sample at the falling edges.
        # Note that the algorithm always finds high for the clock channel.
        edges = np.diff(D[self.clock])
        try:
            indices = np.where(edges == -1)[0]
            # Need to sample the bit on the SR output before the first clock
            edges[2*indices[0] - indices[1]] = -1
            # And remove the last sample because it is what was just shifted in.
            edges[indices[-1]] = 0
            d = [D[i][edges == -1] for i in range(16)]
            return d,D
        except IndexError:
            # occurs for noise triggers, in which case take another trace.
            return dtrace(self,timeout)

def plot_dtrace(name,samples,data,clock,channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
    if clock not in channels:
        channels.append(clock)
    x = np.arange(samples)
    ds = [D[i] for i in range(len(D)) if i in channels]
    ds = [ds[i] + 1.5*i for i in range(len(ds))]
    clock = channels.index(clock)
    for i,d in enumerate(ds):
        plt.plot(x,1.5*i*np.ones(len(x)) - 0.05,linestyle = '--', color = '0.5')
        if i == clock:
            plt.plot(np.linspace(0,samples,2*samples)[:-1],np.array([1,0]*samples)[1:]+(1.5*i),linestyle='steps')                   
        else:
            plt.plot(x,d,linestyle='steps')
    plt.yticks(np.arange(len(channels))*1.5, ['D%i' % i  for i in channels])
    plt.xlim(0,samples-1)
    plt.ylim(-0.1,len(channels)*1.5)
    plt.savefig(name)
    
def plt_csv(*csv_strs):
    fig = plt.figure(figsize=(24,len(arrays)))
    ax = fig.add_subplot(111)
    for i,csv_str in enumerate(csv_strs):
        vals = (csv_str.rstrip('\n')).split(',')
        vals = np.array([(float(val) + (i*1.5)) for val in vals])
        ax.plot(vals,linestyle='steps')
    plt.ylim(-0.1,len(arrays)*1.5 )
    plt.tight_layout()
    fig.show()

def plt_arrays(*arrays):
    fig = plt.figure(figsize=(24,len(arrays)))
    ax = fig.add_subplot(111)
    for i,array in enumerate(arrays):
        ax.plot(array + (i*1.5),linestyle='steps')
    plt.ylim(-0.1,len(arrays)*1.5 )
    plt.tight_layout()
    plt.show()


scope = ScopeInst(0)

# Test of analog read.
#scope.init_analog_channel(1,2)
#times,volts = scope.trace()
#plt.plot(times, volts)
#plt.savefig("test.pdf")

# Test of digital read.
scope.init_digital_channel(clk=0)
#D = scope.dtrace(timeout=30.0)
#print D[8]
traces = [scope.dtrace(timeout=30.0) for i in range(4)]
for trace in traces:
    print trace[0][1]
    print trace[0][2]

"""
outfile = open('register_values.csv','w')
outfile2 = open('raw_values.csv','w')
for trace in traces[1:]:
    tmp = [str(val) for val in trace[0][1]]
    outfile.write(','.join(tmp) + '\n')
    tmp = [str(val) for val in trace[1][1]]
    outfile2.write(','.join(tmp) + '\n')
outfile.close()
outfile2.close()

for i,trace in enumerate(traces[2:]):
    pattern = [0]*176
    pattern[i] = 1
    if np.any(trace[0][1] != pattern):
        tmp = [str(val) for val in pattern]
        print ''.join(tmp)
        tmp = [str(val) for val in trace[0][1]]
        print ''.join(tmp)
        plt_arrays(trace[1][1],trace[1][0])

"""
