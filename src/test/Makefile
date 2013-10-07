HARPLD  = ../harptool -L
HARPAS  = ../harptool -A
HARPEM  = ../harptool -E
HARPDIS = ../harptool -D
4BARCH  = 4b16/16/2

all: simple.bin sieve.bin 2thread.bin simple.4b.bin sieve.4b.bin 2thread.4b.bin bubble.bin bubble.4b.bin dotprod.bin dotprod.4b.bin

run: simple.out sieve.out 2thread.out simple.4b.out sieve.4b.out 2thread.4b.out bubble.out bubble.4b.out dotprod.out dotprod.4b.out

disas: simple.d sieve.d 2thread.d simple.4b.d sieve.4b.d 2thread.4b.d bubble.d bubble.4b.d dotprod.d dotprod.4b.d

%.4b.out : %.4b.bin
	$(HARPEM) -a $(4BARCH) -c $< > $@

%.out : %.bin
	$(HARPEM) -c $< > $@

2thread.bin : boot.HOF lib.HOF 2thread.HOF
	$(HARPLD) -o 2thread.bin $^

2thread.4b.bin : boot.4b.HOF lib.4b.HOF 2thread.4b.HOF
	$(HARPLD) --arch $(4BARCH) -o 2thread.4b.bin $^

bubble.bin : boot.HOF lib.HOF bubble.HOF
	$(HARPLD) -o bubble.bin $^

bubble.4b.bin : boot.4b.HOF lib.4b.HOF bubble.4b.HOF
	$(HARPLD) --arch $(4BARCH) -o bubble.4b.bin $^

simple.bin : boot.HOF lib.HOF simple.HOF
	$(HARPLD) -o $@ $^

sieve.bin : boot.HOF lib.HOF sieve.HOF
	$(HARPLD) -o $@ $^

dotprod.bin : boot.HOF lib.HOF dotprod.HOF
	$(HARPLD) -o $@ $^

simple.4b.bin : boot.4b.HOF lib.4b.HOF simple.4b.HOF
	$(HARPLD) --arch $(4BARCH) -o $@ $^

sieve.4b.bin : boot.4b.HOF lib.4b.HOF sieve.4b.HOF
	$(HARPLD) --arch $(4BARCH) -o $@ $^

dotprod.4b.bin : boot.4b.HOF lib.4b.HOF dotprod.4b.HOF
	$(HARPLD) --arch $(4BARCH) -o $@ $^

%.4b.bin : %.4b.HOF
	$(HARPLD) --arch $(4BARCH) -o $@ $<

%.bin : %.HOF
	$(HARPLD) -o $@ $<

%.4b.HOF : %.s
	$(HARPAS) --arch $(4BARCH) -o $@ $<

%.HOF : %.s
	$(HARPAS) -o $@ $<

%.4b.d : %.4b.HOF
	$(HARPDIS) -o $@ --arch $(4BARCH) $<

%.d : %.HOF
	$(HARPDIS) -o $@ $<

clean:
	rm -f *.HOF *.bin *.out *.d *~
