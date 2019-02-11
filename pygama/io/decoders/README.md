# SIS3302
```
The SIS3302 can produce a waveform from two sources:
    1: ADC raw data buffer: This is a normal digitizer waveform
    2: Energy data buffer: Not sure what this is
Additionally, the ADC raw data buffer can take data in a buffer wrap mode, which seems
to cyclically fill a spot on memory, and it requires that you have to re-order the records
afterwards.
The details of how this header is formatted apparently wasn't important enough for the
SIS engineers to want to put it in the manual, so this is a guess in some places
  0   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ORCA:
      ^^^^ ^^^- ---- ---- ---- ---- ---- ---- most sig bits of num records lost
      ---- ---- ---- ---- ---- ---- ^^^^ ^^^- least sig bits of num records lost
              ^ ^^^- ---- ---- ---- ---- ---- crate
                   ^ ^^^^ ---- ---- ---- ---- card
                          ^^^^ ^^^^ ---- ---- channel
                                            ^ buffer wrap mode
  1   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ORCA: length of waveform (longs)
  2   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ORCA: length of energy   (longs)
  3   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ORCA:
      ^^^^ ^^^^ ^^^^ ^^^^ ---- ---- ---- ---- timestamp[47:32]
                          ^^^^ ^^^^ ^^^^ ^^^^ "event header and ADC ID"
  4   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx SIS:
      ^^^^ ^^^^ ^^^^ ^^^^ ---- ---- ---- ---- timestamp[31:16]
                          ^^^^ ^^^^ ^^^^ ^^^^ timestamp[15:0]
      If the buffer wrap mode is enabled, there are two more words of header:
(5)   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ADC raw data length (longs)
(6)   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ADC raw data start index (longs)
      After this, it will go into the two data buffers directly.
      These buffers are packed 16-bit words, like so:
      xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
      ^^^^ ^^^^ ^^^^ ^^^^ ---- ---- ---- ---- sample N + 1
                          ^^^^ ^^^^ ^^^^ ^^^^ sample N
      here's an example of combining the 16 bit ints to get a 32 bit one.
      print(hex(evt_data_16[-1] << 16 | evt_data_16[-2]))
      The first data buffer is the ADC raw data buffer, which is the usual waveform
      The second is the energy data buffer, which might be the output of the energy filter
      This code should handle arbitrary sizes of both buffers.
      An additional complexity arises if buffer wrap mode is enabled.
      This apparently means the start of the buffer can be anywhere in the buffer, and
      it must be read circularly from that point. Not sure why it is done that way, but
      this should work correctly to disentagle that.
      Finally, there should be a footer of 4 long words at the end:
 -4   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Energy max value
 -3   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx Energy value from first value of energy gate
 -2   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx This word is said to contain "pileup flag, retrigger flag, and trigger counter" in no specified locations...
 -1   1101 1110 1010 1101 1011 1110 1110 1111 Last word is always 0xDEADBEEF
```

# MJDPreampDecoder

```
Decodes the data from a MJDPreamp Object.
Returns:
    adc_val     : A list of floating point voltage values for each channel
    timestamp   : An integer unix timestamp
    enabled     : A list of 0 or 1 values indicating which channels are enabled

Data Format:
0 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
                           ^^^^ ^^^^ ^^^^- device id
1 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  unix time of measurement
2 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  enabled adc mask
3 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  adc chan 0 encoded as a float
4 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  adc chan 1 encoded as a float
....
....
18 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  adc chan 15 encoded as a float
```

# iseg

    Decodes an iSeg HV Card event
    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
    ^^^^ ^^^^ ^^^^ ^^----------------------- Data ID (from header)
    -----------------^^ ^^^^ ^^^^ ^^^^ ^^^^- length
0   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
    ----------^^^^-------------------------- Crate number
    ---------------^^^^--------------------- Card number
1    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx -ON Mask
2    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx -Spare
3    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  time in seconds since Jan 1, 1970
4    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  voltage (chan 0)
5    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  current (chan 0)
6    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  voltage (chan 1)
7    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  current (chan 1)
8    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  voltage (chan 2)
9    xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  current (chan 2)
10   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  voltage (chan 3)
11   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  current (chan 3)
12   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  voltage (chan 4)
13   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  current (chan 4)
14   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  voltage (chan 5)
15   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  current (chan 5)
16   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  voltage (chan 6)
17   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  current (chan 6)
18   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  voltage (chan 7)
19   xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  current (chan 7)
"""
