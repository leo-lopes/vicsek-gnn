================================================================================
                   BLOCO 1 — Refinamento de Fase (η_c)
================================================================================

Este pacote contém tudo para rodar a campanha de refinamento Bloco 1 no cluster.

ESTRUTURA:
  • grid_bloco1a.txt     →  186 jobs (Angular:   η ∈ [0.35, 0.65], L ∈ {32,64,128})
  • grid_bloco1b.txt     →  126 jobs (Vetorial:  η ∈ [0.45, 0.65], L ∈ {32,64,128})
  • bloco1_grids.sh      →  Script gerador (já foi executado)
  • run_grid.sh          →  Idêntico ao Bloco 0 (sem mudanças necessárias)
  • submit.sh            →  Idêntico ao Bloco 0 (genérico para qualquer grid)

================================================================================
                            PASSO A PASSO
================================================================================

1. TRANSFERIR PARA O CLUSTER
   No seu PC:
   $ scp bloco1_grids.sh grid_bloco1a.txt grid_bloco1b.txt leolopes@turing:~/vicsek-gnn/cluster/

2. GERAR AS GRIDS (se já não estão prontas)
   $ cd ~/vicsek-gnn/cluster
   $ bash bloco1_grids.sh

3. SUBMETER BLOCO 1A (Angular)
   $ ./submit.sh grid_bloco1a.txt out/bloco1a

   Espera por resposta:
   [squeue output mostrando jobs 1-186]

4. SUBMETER BLOCO 1B (Vetorial) — pode fazer em paralelo
   $ ./submit.sh grid_bloco1b.txt out/bloco1b

   Espera por resposta:
   [squeue output mostrando jobs 1-126]

5. MONITORAR PROGRESSO
   $ watch -n 15 './status.sh out/bloco1a && echo "---" && ./status.sh out/bloco1b'

   Ou separado:
   $ ./status.sh out/bloco1a
   $ ./status.sh out/bloco1b

================================================================================
                          DETALHES DOS JOBS
================================================================================

BLOCO 1A — Angular (n=0)
  η:   0.35, 0.36, ..., 0.64, 0.65  [31 valores, step 0.01]
  L:   128, 64, 32  [ordem decrescente → jobs longos na fila primeiro]
  Replicas: 2 por (η, L)
  Duração:  ~3–4 horas total (L=128 ~20–30 min por job, L=64 ~10–15 min, L=32 ~5–8 min)
  Total:    186 jobs = 31 η × 3 L × 2 replicas

BLOCO 1B — Vetorial (n=1)
  η:   0.45, 0.46, ..., 0.64, 0.65  [21 valores, step 0.01]
  L:   128, 64, 32  [ordem decrescente]
  Replicas: 2 por (η, L)
  Duração:  ~2–3 horas total (mesma escala de duração que Angular)
  Total:    126 jobs = 21 η × 3 L × 2 replicas

Sementes: 241–560 (continuação de Bloco 0, que usou 1–240)

================================================================================
                        ANÁLISE PÓS-CONCLUSÃO
================================================================================

Quando ambos os blocos terminarem (100% completo):

1. Extrair dados de susceptibilidade:
   $ ./phi.sh out/bloco1a > phi_bloco1a.dat
   $ ./phi.sh out/bloco1b > phi_bloco1b.dat

2. Identificar máximos refinados:
   $ sort -k7 -gr phi_bloco1a.dat | head -5  # Top 5 picos Angular
   $ sort -k7 -gr phi_bloco1b.dat | head -5  # Top 5 picos Vetorial

3. Determinar η_c com precisão:
   • Esquerda do pico: η onde χ começa a subir
   • Direita do pico: η onde χ cai para ~50% do máximo
   • η_c ≈ (eta_left + eta_right) / 2

Esses novos η_c servem como entrada para BLOCO 2 (geração de dados de treino para GNN).

================================================================================
                            TROUBLESHOOTING
================================================================================

❌ "error: sbatch: command not found"
   → SLURM não está instalado; você está no login node, não no cluster compute
   → Verifique: hostname (deve ser "turing")

❌ "error: OUT directory not found"
   → submit.sh falhou a criar out/bloco1a
   → Verifique permissões: ls -la ~/vicsek-gnn/cluster/out/

❌ "FV not found"
   → Binary não foi compilado no cluster
   → Certifique-se: ~/vicsek-gnn/sim/FV existe e é executável
   → Se não: cd ~/vicsek-gnn/sim && make clean && make

✓ Tudo ok?
   → Veja: squeue -u leolopes (jobs rodando/esperando)
   → Veja: ls -la out/bloco1a/.done/ (sentinelas completadas)

================================================================================
