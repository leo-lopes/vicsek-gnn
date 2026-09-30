"""Mini-campanha para rodar no PC enquanto o cluster esta fora do ar.

Uso (dentro do WSL/Linux, na pasta pc/):
    python3 pc_campaign.py              # mede a velocidade, mostra a estimativa e pergunta
    python3 pc_campaign.py -j 4 --yes   # 4 jobs em paralelo, sem perguntar
    python3 pc_campaign.py --so piloto  # so um dos grupos: bloco0, piloto, treino

Pode interromper (Ctrl+C) e rodar de novo: jobs com meta "done" sao pulados.
Sementes 900001+ nao colidem com as do cluster, entao os resultados podem ser
juntados depois aos do bloco 0 do cluster como sementes extras.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FV = os.path.join(HERE, "..", "sim", "FV")
OUT = os.path.join(HERE, "pc_out")
V0 = 0.5


def grid():
    jobs = []  # (grupo, argumentos)
    # Bloco 0 reduzido: L = 64 e 128, 2 sementes (o cluster completa L = 256 e mais sementes)
    for L in (64, 128):
        for eta in np.round(np.arange(0.30, 0.6001, 0.02), 3):
            for _ in range(2):
                jobs.append(("bloco0", f"-L {L} -r 2 -v {V0} -n 0 -e {eta} -t 50000 -T 200000"))
        for eta in np.round(np.arange(0.50, 0.7001, 0.01), 3):
            for _ in range(2):
                jobs.append(("bloco0", f"-L {L} -r 2 -v {V0} -n 1 -G 1 -e {eta} -t 50000 -T 200000"))
    # Piloto das bandas: caixa 512x64, inicio alinhado, 1 semente, 7 densidades
    for noise, eta in (("-n 0", 0.40), ("-n 1 -G 1", 0.55)):
        for rho in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            jobs.append(("piloto", f"-L 512 -Y 64 -r {rho} -v {V0} {noise} -e {eta} -i 1"
                                   f" -t 100000 -T 200000 -C 1000 -c 4"))
    # Dados de treino piloto (para desenvolver a GNN na GTX 1650): 3 sementes por ruido
    for noise, eta in (("-n 0", 0.40), ("-n 1 -G 1", 0.55)):
        for _ in range(3):
            jobs.append(("treino", f"-L 128 -r 2 -v {V0} {noise} -e {eta} -i 1 -t 100000 -T 100000 -S 1000"))
    return [(g, f"{a} -s {900000 + k}") for k, (g, a) in enumerate(jobs, start=1)]


def opt(args, flag, cast=str, default=None):
    t = args.split()
    return cast(t[t.index(flag) + 1]) if flag in t else default


def tag(args):
    n = opt(args, "-n", int, 0)
    g = opt(args, "-G", int, 0)
    L = opt(args, "-L", int)
    Y = opt(args, "-Y", int, L)
    suf = ("G" if g else "o") if n == 1 else ""
    return (f"n{n}{suf}_v{opt(args, '-v', float, 0.5):.3f}_r{opt(args, '-r', float):.3f}"
            f"_e{opt(args, '-e', float):.4f}_L{L}x{Y}_s{opt(args, '-s', int)}")


def cost(args):
    """particulas x passos (para a estimativa de tempo)."""
    L = opt(args, "-L", int)
    Y = opt(args, "-Y", int, L)
    return opt(args, "-r", float) * L * Y * (opt(args, "-t", int, 0) + opt(args, "-T", int))


def done(grupo, args):
    p = os.path.join(OUT, grupo, f"meta_{tag(args)}.json")
    try:
        return json.load(open(p)).get("status") == "done"
    except (OSError, ValueError):
        return False


def benchmark():
    with tempfile.TemporaryDirectory() as d:
        t0 = time.time()
        subprocess.run([FV, "-L", "128", "-r", "2", "-e", "0.4", "-s", "1", "-T", "400", "-o", d, "-z"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return (time.time() - t0) / (2 * 128 * 128 * 400)


def run(grupo, args):
    d = os.path.join(OUT, grupo)
    os.makedirs(d, exist_ok=True)
    r = subprocess.run([FV] + args.split() + ["-o", d], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return grupo, args, r.returncode, r.stderr.decode()[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="jobs em paralelo")
    ap.add_argument("--so", choices=["bloco0", "piloto", "treino"], help="rodar so um grupo")
    ap.add_argument("--yes", action="store_true", help="nao perguntar antes de comecar")
    ap.add_argument("--quick", action="store_true", help="teste: divide os passos por 1000")
    a = ap.parse_args()

    if not os.access(FV, os.X_OK):
        subprocess.run(["make", "-C", os.path.join(HERE, "..", "sim")], check=True)
    jobs = [(g, s) for g, s in grid() if a.so in (None, g)]
    if a.quick:
        def shrink(s):
            t = s.split()
            for f in ("-t", "-T"):
                if f in t:
                    t[t.index(f) + 1] = str(max(1, int(t[t.index(f) + 1]) // 1000))
            if "-S" in t:
                t[t.index("-S") + 1] = "10"
            if "-C" in t:
                t[t.index("-C") + 1] = "10"
            return " ".join(t)
        jobs = [(g, shrink(s)) for g, s in jobs]
    todo = [(g, s) for g, s in jobs if not done(g, s)]

    sec = benchmark()
    print(f"Velocidade medida: {sec * 1e6:.3f} us por particula por passo")
    total_h = 0.0
    for g in ("bloco0", "piloto", "treino"):
        sel = [s for gg, s in todo if gg == g]
        if sel:
            h = sum(cost(s) for s in sel) * sec / 3600
            total_h += h
            print(f"  {g:7s}: {len(sel):3d} jobs, ~{h:6.1f} CPU-h")
    maior = max((cost(s) * sec / 3600 for _, s in todo), default=0)
    parede = max(total_h / a.j, maior)
    print(f"Total: {len(todo)} jobs (de {len(jobs)}), ~{total_h:.1f} CPU-h;"
          f" com {a.j} jobs em paralelo, ~{parede:.1f} h de relogio")
    if not todo:
        return
    if not a.yes and input("Comecar? [s/N] ").strip().lower() not in ("s", "sim", "y", "yes"):
        return

    # jobs mais longos primeiro, para o fim nao ficar esperando um job grande sozinho
    todo.sort(key=lambda gs: -cost(gs[1]))
    t0, feitos = time.time(), 0
    with ThreadPoolExecutor(max_workers=a.j) as ex:
        futs = [ex.submit(run, g, s) for g, s in todo]
        for f in as_completed(futs):
            g, s, rc, err = f.result()
            feitos += 1
            estado = "ok" if rc == 0 else f"ERRO ({rc}): {err}"
            print(f"[{feitos}/{len(todo)}] {time.time() - t0:7.0f} s  {g}: {tag(s)}  {estado}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
