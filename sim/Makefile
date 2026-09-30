CC ?= cc
# NAO use -march=native aqui: no Turing o no de login tem CPU mais nova que os
# nos de calculo, entao o binario morre com SIGILL (exit 132) dentro do SLURM.
# Os proprios nos de calculo sao heterogeneos, entao nem compilar num deles salva.
CFLAGS ?= -O3 -mtune=generic
# No seu PC, onde roda na mesma maquina que compila:
#   make CFLAGS="-O3 -march=native"

FV: FastVicsek.c cokus3.c
	$(CC) $(CFLAGS) -o FV FastVicsek.c cokus3.c -lm

test: FV
	rm -rf teste && mkdir -p teste
	./FV -L 48 -r 2 -v 0.5 -n 0 -e 0.35 -s 7 -t 2000 -T 3000 -S 300 -o teste -z
	./FV -L 48 -r 2 -v 0.5 -n 1 -G 1 -e 0.55 -s 8 -t 2000 -T 3000 -S 300 -o teste -z
	./FV -L 48 -r 2 -v 0.5 -n 1 -G 0 -e 0.45 -s 9 -t 2000 -T 3000 -S 300 -o teste -z
	./FV -L 48 -r 2 -v 0.5 -n 2 -e 0.35 -s 10 -t 2000 -T 3000 -S 300 -o teste -z
	python3 ../python/validate.py teste

clean:
	rm -rf FV teste

.PHONY: test clean
