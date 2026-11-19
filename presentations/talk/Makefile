all: talk_60min.pdf talk_30min.pdf
%.pdf: %.tex preamble.tex sec_*.tex
	latexmk -pdf -interaction=nonstopmode $<
clean:
	latexmk -C
