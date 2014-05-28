import visa
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack


THRESHOLD = 0.75

# Utilities
def binary_list(x):
    return binary_list(x/2) + [x%2] if x > 1 else [x]


def filled_binary_list(x,nbytes=4):
    vals = binary_list(x)
    return [0]*((nbytes*8) - len(vals)) + vals


# Scope class, which runs the MSO4104 Oscilloscope
class ScopeInst(visa.Instrument):
    def __init__(self,USB_num=0,GetInstId=0):
        self.InstId = ""
        self.dwidth = 1
        self.clock = 0
        available = [ inst for inst in visa.get_instruments_list() if ("USB%i" % USB_num) in inst]
        if len(available) == 0:
            print "No USB%i instrument found." % USB_num
            sys.exit(1)
        visa.Instrument.__init__(self,available[0])
        if GetInstId == 1:
            self.write("*IDN?")
            self.InstId=self.read()

    def sendStr(self,strIn):
        self.write(strIn)

    def readStr(self):
        return self.read()

    def init_analog_channel(self,channel,data_width=1):
        self.dwidth = data_width
        self.write('DATA:SOU CH%i' % channel)
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
        while '1' in self.ask("ACQ:STATE?"):
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

    def init_fourier_transform(self, channel, data_width=1):
        self.dwidth = data_width
        self.write('MATH:TYPE FFT')
        self.write('MATH:DEFINE "FFT( CH%i )"' % channel)
        self.write('MATH:SPECT:WIN HAN')
        self.write('MATH:SPECT:MAG DB')
        self.write('DATA:SOU MATH')
        self.write('DATA:WIDTH %i' % self.dwidth)   
        self.write('DATA:ENC FAS')

    def ftrace(self,timeout=30.0):
        ymult = float(self.ask('MATH:VERT:SCA?'))
        yzero = float(self.ask('MATH:VERT:POS?'))
        xmult = float(self.ask('MATH:HOR:SCA?'))
        # Ask for single trigger and wait for it, to ensure a single waveform.
        self.write('ACQ:STOPA SEQ')
        self.write('ACQ:STATE RUN')
        startt = time.time()
        while '1' in self.ask("ACQ:STATE?"):
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
        db = vals * ymult  + yzero
        freq = np.arange(len(db)) * xmult
        return freq,db

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
        while '1' in self.ask("ACQ:STATE?"):
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
        indices = np.where(edges == -1)[0]
        if len(indices) == 0: # timeout
            return [],[]
        if len(indices) == 1: # misfire, so resample
            return dtrace(self,timeout)
        # Need to sample the bit on the SR output before the first clock
        edges[2*indices[0] - indices[1]] = -1
        # And remove the last sample because it is what was just shifted in.
        edges[indices[-1]] = 0
        d = [D[i][edges == -1] for i in range(16)]
        return d,D



# Generic plotting functions.
def plt_dtrace(data,clock,channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],savename=''):
    nsamples = len(data[clock])+1
    if clock not in channels:
        channels.append(clock)
    x = np.arange(nsamples)
    ds = [data[i] for i in range(len(data)) if i in channels]
    ds = [ds[i] + 1.5*i for i in range(len(ds))]
    ds = [np.insert(d,0,0) for d in ds] 
    clock = channels.index(clock)
    fig = plt.figure(figsize=(24,len(ds)))
    ax = fig.add_subplot(111)
    for i,d in enumerate(ds):
        ax.plot(x,1.5*i*np.ones(len(x)) - 0.05,linestyle = '--', color = '0.5')
        if i == clock:
            ax.plot(np.linspace(0,nsamples,2*nsamples)[:-1],np.array([1,0]*nsamples)[1:]+(1.5*i),linestyle='steps')                   
        else:
            ax.plot(x,d,linestyle='steps')
    plt.yticks(np.arange(len(channels))*1.5, ['D%i' % i  for i in channels])
    plt.ylim(-0.1,len(channels)*1.5)
    plt.xlim(0,nsamples-1)
    plt.tight_layout()
    if savename == '':
        plt.show()
    else:
        plt.savefig(savename)

    
def plt_csv(*csv_strs):
    fig = plt.figure(figsize=(24,len(arrays)))
    ax = fig.add_subplot(111)
    for i,csv_str in enumerate(csv_strs):
        vals = (csv_str.rstrip('\n')).split(',')
        vals = np.array([(float(val) + (i*1.5)) for val in vals])
        ax.plot(vals,linestyle='steps')
    plt.ylim(-0.1,len(arrays)*1.5 )
    plt.tight_layout()
    plt.show()


def plt_arrays(*arrays):
    fig = plt.figure(figsize=(24,len(arrays)))
    ax = fig.add_subplot(111)
    for i,array in enumerate(arrays):
        ax.plot(array + (i*1.5),linestyle='steps')
    plt.ylim(-0.1,len(arrays)*1.5 )
    plt.tight_layout()
    plt.show()

def plt_ftrace(trace, savename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(trace[0], trace[1], 'r-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)

# Quick example program, takes and displays a single fourier trace.
def main():
    scope = ScopeInst(0)
    scope.init_fourier_transform(1)
    traces  = [scope.ftrace(timeout=.1) for i in xrange(100)]
    average = sum((trace[1] for trace in traces))/ 100.0
    trace = traces[0][0], average
    plt_ftrace(trace)

if __name__ == "__main__":
    main()



