Summary:
The new code is now completely separate from the original, and does not 
import or use any of the original code.
The original code for the LT3MAPS testing is now stored in LT3MAPS_BACKUP.

The new code is commented so that the python builtin help() function will
give useful results. The main modules are pix.py, which handles making and
storing measurements, and chip.py, which handles writing to the chip.

-------------------------------------------------------------------------------

Where's the old code?

The current code is sufficiently different from the original in the way
it handles the testing devices (in particular the data generator), that 
it can be dangerous (in the sense of producing undefined behavior) to mix 
the two testing codes without knowing what the state of the system was
at the end of the last test. 

So, to prevent that happening on accident, I'm going to completely separate 
the new code from the original, and no longer import any of the original 
modules. 

-------------------------------------------------------------------------------

How does chip.py work?

chip.py uses a small extension of the visa GpibInstrument class to setup and
communicate with the three devices (data generator, signal generator, counter).
It also has a more complicated driver class which sets up the generator so that
it can write long instructions as quickly as possibly. 

Typical setup is:
import chip
...
chip.init_hpgene(chip.hpgene) # setup signal generator
chip.init_hpcntr(chip.hpcntr) # setup counter
driver = chip.DgeneDriver(chip.dgene, number_instructions=1 or 4, 
                          config_size=800 or 380) 
driver.init_blocks()
...
make measurements using commands from driver and hpcntr and hpgene
...

To see more specific help information for the chip module, try (in a python 
command line):
import chip
help(chip)
or look at the comments in the code.

-------------------------------------------------------------------------------

What does pix do?

pix.py has several functions that I have written to make specific measurements
of the chip. They are individually commented. pix.py also defines a few
systems to store (write to and read from csv files) relevant pixel data, using
the PixelLibrary class.

pix.py also has a state saving system (which uses the file state.dat, so don't
delete it), so that it can record some information about the state of the 
pixels. Such as whether they are tuned and whether they are hit-enabled. This 
speeds up many operations by skipping unnecessary steps, and also saves me 
from having to remember. Note that this state system will only be reliable if 
every function which changes the state also records what it changes.

Again, to see more specific help information for the chip module, try:
import pix
help(pix)

-------------------------------------------------------------------------------

