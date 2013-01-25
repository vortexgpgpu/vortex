FIGS = fig/simd_core.pdf fig/byte_enc.pdf fig/word_enc.pdf
DIRS = fig

NAME = harp_iset

all: $(NAME).pdf

$(NAME).pdf: $(NAME).tex $(FIGS)

%.pdf: %.tex
	pdflatex $<

fig/%.pdf: fig/%.fig
	cd fig; $(MAKE) $(MFLAGS) $(@F); cd ..

clean:
	rm -f $(NAME).pdf $(NAME).aux $(NAME).log $(NAME).toc $(NAME).snm \
              $(NAME).out $(NAME).nav fig/*.pdf

force_look:
	true
