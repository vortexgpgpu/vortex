HARPLD = ../harptool -L
HARPAS = ../harptool -A
4WARCH = 2b16/1/8

all: sieve.bin 2thread.bin sieve.4w.bin 2thread.4w.bin

2thread.bin : lib.HOF 2thread.HOF boot.HOF
	$(HARPLD) -o 2thread.bin boot.HOF 2thread.HOF lib.HOF

2thread.4w.bin : lib.4w.HOF 2thread.4w.HOF boot.4w.HOF
	$(HARPLD) --arch $(4WARCH) -o 2thread.4w.bin boot.4w.HOF 2thread.4w.HOF 2thread.4w.HOF lib.4w.HOF

sieve.bin : lib.HOF sieve.HOF boot.HOF
	$(HARPLD) -o sieve.bin boot.HOF sieve.HOF lib.HOF

sieve.4w.bin : lib.4w.HOF sieve.4w.HOF boot.4w.HOF
	$(HARPLD) --arch $(4WARCH) -o sieve.4w.bin boot.4w.HOF sieve.4w.HOF lib.4w.HOF

%.4w.bin : %.4w.HOF
	$(HARPLD) --arch $(4WARCH) -o $@ $<

%.bin : %.HOF
	$(HARPLD) -o $@ $<

%.4w.HOF : %.s
	$(HARPAS) --arch $(4WARCH) -o $@ $<

%.HOF : %.s
	$(HARPAS) -o $@ $<

clean:
	rm -f *.HOF *.bin *~
