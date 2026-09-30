# Simulador FastVicsek e formato dos dados

Simulador em C do modelo de Vicsek usado no projeto "Aprendizagem profunda de dinâmica coletiva a partir do modelo de Vicsek": opções, convenções e formato dos arquivos de saída. A visão geral do projeto está no README principal.

```
vicsek-gnn/
├── README.md
├── sim/
│   ├── FastVicsek.c      Vicsek puro (N constante), derivado do FastBoid-Malthusian
│   ├── cokus3.c          gerador Mersenne Twister (inalterado)
│   └── Makefile          make / make test
├── python/
│   ├── vicsek_io.py      leitura dos binários, vizinhança periódica, densidades exatas do ruído
│   ├── validate.py       testes: vizinhança, regra de alinhamento e distribuição exata do ruído
│   └── theory_noise.py   kappa ótimo, NLL do oráculo e custo de usar von Mises
├── cluster/
│   ├── make_grids.sh     grades originais dos blocos 0 a 4
│   └── run_grid.sh       job array SLURM: uma linha da grade por tarefa
└── pc/
    └── pc_campaign.py    mini-campanha para o PC (retomável, com estimativa de tempo)
```

## Compilar e testar

```bash
cd sim
make          # gera ./FV (sem -march=native: os nós do cluster são heterogêneos; veja o Makefile)
make test     # 4 simulações curtas + python/validate.py (precisa de numpy e scipy)
./FV -h       # todas as opções
```

## Exemplos

```bash
# ruído angular, 2e4 passos de transiente, 1e5 de produção, snapshot de treino a cada 1000
./FV -L 256 -r 2 -e 0.40 -n 0 -s 17 -t 20000 -T 100000 -S 1000 -o out/

# ruído vetorial padrão (eta*n), caixa alongada 1024x128, início alinhado em x, campos coarse-grained
./FV -L 1024 -Y 128 -r 1 -e 0.55 -n 1 -G 1 -i 1 -s 7 -t 100000 -T 300000 -C 1000 -c 4 -o out/

# controle "oráculo-VM": mesma regra, ruído von Mises com o mesmo <cos> do uniforme de eta = 0.40
./FV -L 256 -r 2 -e 0.40 -n 2 -s 3 -T 100000 -o out/

# reiniciar de um checkpoint (as saídas ganham o sufixo _c<t>, nada é sobrescrito)
./FV -L 256 -r 2 -e 0.40 -n 0 -s 17 -T 100000 -i 2 -f out/chk_n0_v0.500_r2.000_e0.4000_L256x256_s17.bin -o out/
```

O diretório de saída precisa existir.

## Convenções

- Raio de interação R = 1 = lado da caixa da lista de células; Δt = 1. Mudar R equivale a reescalar
  v0 → v0/R, ρ → ρR², L → L/R.
- Atualização: θ(t+1) a partir do estado em t; depois x(t+1) = x(t) + v0 (cos θ(t+1), sin θ(t+1)), com PBC.
- A própria partícula entra na média (n conta a própria partícula), como no código original.
- Ruído (`-n`):
  - `0` angular: θ(t+1) = θ̄ + U[−ηπ, ηπ], com η ∈ [0, 1];
  - `1` vetorial: θ(t+1) = arg(Σ v_j + A ξ), ξ vetor unitário aleatório, A = η(n+1) com `-G 0`
    (como no código original) ou A = η·n com `-G 1` (Grégoire & Chaté 2004; Chaté et al. 2008);
  - `2` angular von Mises: θ̄ + VM(0, κ); κ padrão casa ⟨cos⟩ com o uniforme de mesmo η (ou use `-k`).
- Sementes: `-s` explícita (use o índice do job). Internamente, seedMT(2s+1). Sem `-s`, usa tempo+pid.

## Saídas (`tag` = n{ruído}{G|o}_v{v0}_r{ρ}_e{η}_L{lx}x{ly}_s{semente})

| Arquivo | Conteúdo |
| --- | --- |
| `ts_<tag>.dat.gz` | t, Φ, direção média, desvio relativo da ocupação das caixas (a cada `-w` passos) |
| `snap_<tag>.bin` | registros de treino (a cada `-S` passos) |
| `cg_den/sop/vop_<tag>.bin` | densidade, ordem local e ângulo coarse-grained (a cada `-C` passos, caixa `-c`) |
| `chk_<tag>.bin` | checkpoint com ids, gravado no fim (e a cada `-P` passos) |
| `meta_<tag>.json` | parâmetros, semente, status (`running`/`done`) e tempo de CPU |

Formato binário (little-endian). Cabeçalho de 128 bytes:
`char magic[8]; int32 version, N, lx, ly, noise, gcnorm; int64 t; uint64 seed; double eta, rho, v0, kappa;` e zeros.

- Snapshot (`VSKSNAP1`), `t` = tempo do estado de entrada:
  `int32 id[N]; double x[N], y[N]; float th0[N], th1[N], thbar[N], C[N]; int32 n[N]`
  (th0 = θ(t), th1 = θ(t+1), thbar = direção média antes do ruído, C = |Σv|/n, n = vizinhos + 1).
  São 40 bytes por partícula.
- Checkpoint (`VSKCHKP1`), `t` = tempo acumulado: `int32 id[N]; double x[N], y[N], vx[N], vy[N]`.
- Campos coarse-grained: `int32 lx/c, ly/c` e depois quadros `float32[ly/c][lx/c]`.
  Em caixas vazias, ordem e ângulo valem 0: use densidade 0 como máscara.

Leitura em Python:

```python
import vicsek_io as vio
for s in vio.read_snapshots("out/snap_<tag>.bin"):
    print(s.header["t"], s.x.shape, s.th1[:3])
thbar, C, n = vio.vicsek_mean(s)          # recalcula a regra a partir de (x, y, th0)
```

## Diferenças em relação ao FastBoid-Malthusian-birth_mode.c

- Sem nascimento e morte: saem LAMBDA, LD, BIRTH_MODE, o vetor `copy` e os `realloc`.
- `seedMT` força semente ímpar, então 2k e 2k+1 geravam a mesma simulação: agora usa 2s+1.
- `seedMT2` usava `time(NULL)`: jobs iniciados no mesmo segundo repetiam a semente.
- Nomes de arquivo sem semente: réplicas em paralelo se sobrescreviam.
- Coarse-graining dividia por zero em caixas vazias (NaN).
- `x + L` arredondado para exatamente `L` podia gerar índice de caixa fora do vetor.
- Progresso sai no stderr só com `-q` (antes: um `printf` por passo no stdout).
- Com λ = 0, o original e o novo dão o mesmo ⟨Φ⟩ dentro do erro (L = 32, ruído vetorial η(n+1)):
  0,8555 ± 0,0008 contra 0,8547 ± 0,0007 em η = 0,45; 0,7604 ± 0,0014 contra 0,7594 ± 0,0004 em η = 0,55.

## Campanha

Os scripts que rodam no cluster estão em `../cluster/` (`submit.sh <bloco>`, `status.sh`, `phi.sh`) e a mini-campanha para o PC está em `../pc/pc_campaign.py`. O README principal mostra a ordem das etapas.
