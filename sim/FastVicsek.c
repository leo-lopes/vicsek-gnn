/* FastVicsek.c -- modelo de Vicsek puro (N constante), com saida para treino de GNN.
 *
 * Derivado de FastBoid-Malthusian-birth_mode.c. O que mudou:
 *   - removida a dinamica de nascimento/morte (LAMBDA, LD, BIRTH_MODE, vetor copy, reallocs);
 *   - todos os parametros vem da linha de comando (getopt), inclusive a semente;
 *   - semente explicita: seedMT(2*s+1). O seedMT do cokus3 forca a semente a ser impar,
 *     entao 2k e 2k+1 davam a MESMA sequencia; e seedMT2() usa time(NULL), que repete a
 *     semente para jobs que comecam no mesmo segundo no cluster;
 *   - nomes de arquivo incluem a semente (jobs em paralelo nao se sobrescrevem);
 *   - ids persistentes (as particulas sao reordenadas por caixa a cada passo);
 *   - snapshots binarios para treino: por particula, (x_t, y_t, th_t, th_{t+1}, thbar_t, C_t, n_t);
 *   - ruido: 0 = angular uniforme em [-eta*pi, eta*pi]; 1 = vetorial; 2 = angular von Mises
 *     (controle "oraculo-VM": mesma regra, so troca a forma do ruido);
 *   - ruido vetorial: -G 0 usa amplitude eta*(n_i+1), como no codigo original;
 *                     -G 1 usa eta*n_i, como em Gregoire & Chate (2004) e Chate et al. (2008);
 *     (n_i conta a propria particula);
 *   - condicao periodica robusta a arredondamento (x == lx nunca acontece);
 *   - campos coarse-grained sem divisao por zero em caixas vazias;
 *   - checkpoint binario (com ids) para reiniciar (-i 2 -f arquivo).
 *
 * Raio de interacao = 1 = tamanho da caixa (unidade de comprimento). Mudar R equivale a
 * reescalar v0 -> v0/R, rho -> rho*R^2, L -> L/R.
 *
 * Compilar:  cc -O3 -march=native -o FV FastVicsek.c cokus3.c -lm
 * Exemplo :  ./FV -L 128 -e 0.40 -r 2 -v 0.5 -n 0 -s 17 -t 20000 -T 100000 -S 1000 -o out/
 * Ajuda   :  ./FV -h
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

extern void seedMT(unsigned long seed);
extern double ranMT(void);

/* ------------------------------------------------------------------ parametros */
static int    lx = 128, ly = -1;
static double eta = 0.30, rho = 2.0, v0 = 0.5, kappa = -1.0;
static int    noise = 0;          /* 0 angular uniforme, 1 vetorial, 2 angular von Mises     */
static int    gcnorm = 0;         /* so para noise=1: 0 -> eta*(n+1), 1 -> eta*n            */
static long   tmax = 100000;      /* passos de producao                                       */
static long   t0 = 0;             /* transiente (sem saida)                                   */
static int    init = 0;           /* 0 aleatorio, 1 alinhado em x, 2 le checkpoint (-f)       */
static char   initfile[1024] = "";
static char   outdir[1024] = "./";
static unsigned long seed = 0;
static int    seed_given = 0;
static long   ts_every = 1;       /* serie temporal de Phi                                    */
static long   snap_every = 0;     /* snapshots de treino (0 = desligado)                      */
static long   snap_start = 1;     /* primeiro passo de producao com snapshot                  */
static long   cg_every = 0;       /* campos coarse-grained (0 = desligado)                    */
static int    cg_size = 1;
static long   chk_every = 0;      /* checkpoint intermediario (sempre grava um no final)      */
static long   movie_every = 0;    /* matriz de densidade no stdout (DynamicLattice)           */
static int    verbose = 0;

/* ------------------------------------------------------------------ estado */
static int      n;                               /* numero de particulas (constante)      */
static double  *x, *y, *vx, *vy;
static int32_t *pid;                             /* id persistente                        */
static double  *xtmp, *ytmp, *vxtmp, *vytmp;
static int32_t *pidtmp;
static int     *box, *boxn, *boxs, *boxt, nbox;
static double  *vxint, *vyint;
static int     *nint;
static float   *s_th0, *s_th1, *s_thbar, *s_cloc; /* buffers de snapshot                 */
static int32_t *s_nn;
static char     tag[512];
static FILE    *fsnap = NULL;
static long     tchk = 0;                        /* tempo gravado no checkpoint lido (-i 2) */

/* ------------------------------------------------------------------ utilidades */
static void *xmalloc(size_t sz)
{
  void *p = malloc(sz);
  if (!p && sz) { fprintf(stderr, "sem memoria (%zu bytes)\n", sz); exit(1); }
  return p;
}

static double wrap(double a, double l)          /* PBC robusta: resultado em [0, l) */
{
  if (a < 0.0)  a += l;
  if (a >= l)   a -= l;                          /* pega o caso -1e-17 + l == l      */
  if (a < 0.0 || a >= l) a = fmod(fmod(a, l) + l, l);
  if (a >= l)   a = 0.0;
  return a;
}

/* I1(k)/I0(k) por fracao continuada (estavel para todo k >= 0) */
static double BesselRatio(double k)
{
  int j, m;
  double r = 0.0;
  if (k < 1e-12) return 0.5 * k;
  m = 60 + (int) (2.0 * k);
  if (m > 20000) m = 20000;
  for (j = m; j >= 1; --j) r = 1.0 / (2.0 * j / k + r);
  return r;
}

/* kappa da von Mises com o mesmo <cos> do ruido uniforme em [-eta*pi, eta*pi] */
static double KappaFromEta(double e)
{
  double target, lo = 0.0, hi = 1e6, mid;
  int it;
  if (e <= 0.0) return 1e6;
  if (e >= 1.0) return 0.0;
  target = sin(PI * e) / (PI * e);
  for (it = 0; it < 200; ++it) {
    mid = 0.5 * (lo + hi);
    if (BesselRatio(mid) < target) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/* amostra von Mises(0, k): Best & Fisher (1979) */
static double VonMises(double k)
{
  double tau, rr, r, u1, u2, z, f, c;
  if (k < 1e-8) return PI * (2.0 * ranMT() - 1.0);
  tau = 1.0 + sqrt(1.0 + 4.0 * k * k);
  rr  = (tau - sqrt(2.0 * tau)) / (2.0 * k);
  r   = (1.0 + rr * rr) / (2.0 * rr);
  for (;;) {
    u1 = ranMT();
    u2 = ranMT();
    z  = cos(PI * u1);
    f  = (1.0 + r * z) / (r + z);
    c  = k * (r - f);
    if (c * (2.0 - c) - u2 > 0.0) break;
    if (u2 > 0.0 && log(c / u2) + 1.0 - c >= 0.0) break;
  }
  if (f > 1.0) f = 1.0;
  if (f < -1.0) f = -1.0;
  return (ranMT() > 0.5) ? acos(f) : -acos(f);
}

/* ------------------------------------------------------------------ E/S binaria
   Cabecalho de 128 bytes (little-endian):
     char magic[8]; int32 version, N, lx, ly, noise, gcnorm; int64 t; uint64 seed;
     double eta, rho, v0, kappa; zeros ate 128 bytes.
   Snapshot ("VSKSNAP1"), t = tempo do estado de entrada; depois do cabecalho:
     int32 id[N]; double x[N], y[N]; float th0[N], th1[N], thbar[N], C[N]; int32 n[N]
   Checkpoint ("VSKCHKP1"): int32 id[N]; double x[N], y[N], vx[N], vy[N]            */
static void WriteHeader(FILE *fp, const char *magic, long t)
{
  unsigned char h[128];
  int32_t iv[6];
  int64_t tt = t;
  uint64_t sd = seed;
  double dv[4];
  memset(h, 0, sizeof(h));
  memcpy(h, magic, 8);
  iv[0] = 1; iv[1] = n; iv[2] = lx; iv[3] = ly; iv[4] = noise; iv[5] = gcnorm;
  memcpy(h + 8, iv, sizeof(iv));
  memcpy(h + 32, &tt, 8);
  memcpy(h + 40, &sd, 8);
  dv[0] = eta; dv[1] = rho; dv[2] = v0; dv[3] = kappa;
  memcpy(h + 48, dv, sizeof(dv));
  if (fwrite(h, 1, 128, fp) != 128) { fprintf(stderr, "erro de escrita\n"); exit(1); }
}

static void WriteCheckpoint(long t)
{
  char name[1600], tmpname[1700];
  FILE *fp;
  snprintf(name, sizeof(name), "%schk_%s.bin", outdir, tag);
  snprintf(tmpname, sizeof(tmpname), "%s.tmp", name);
  fp = fopen(tmpname, "wb");
  if (!fp) { fprintf(stderr, "nao consegui abrir %s\n", tmpname); exit(1); }
  WriteHeader(fp, "VSKCHKP1", t);
  fwrite(pid, sizeof(int32_t), n, fp);
  fwrite(x, sizeof(double), n, fp);
  fwrite(y, sizeof(double), n, fp);
  fwrite(vx, sizeof(double), n, fp);
  fwrite(vy, sizeof(double), n, fp);
  fclose(fp);
  rename(tmpname, name);                          /* troca atomica: nunca fica pela metade */
}

static void WriteSnapshot(long tin)
{
  WriteHeader(fsnap, "VSKSNAP1", tin);
  fwrite(pid, sizeof(int32_t), n, fsnap);
  fwrite(x, sizeof(double), n, fsnap);
  fwrite(y, sizeof(double), n, fsnap);
  fwrite(s_th0, sizeof(float), n, fsnap);
  fwrite(s_th1, sizeof(float), n, fsnap);
  fwrite(s_thbar, sizeof(float), n, fsnap);
  fwrite(s_cloc, sizeof(float), n, fsnap);
  fwrite(s_nn, sizeof(int32_t), n, fsnap);
  fflush(fsnap);
}

/* ------------------------------------------------------------------ condicao inicial */
static void AllocState(void)
{
  x = xmalloc(n * sizeof(double));   y = xmalloc(n * sizeof(double));
  vx = xmalloc(n * sizeof(double));  vy = xmalloc(n * sizeof(double));
  pid = xmalloc(n * sizeof(int32_t));
}

static void InitialCondition(void)
{
  int h;
  double th;
  if (init < 2) {
    n = (int) (rho * lx * ly + 0.000001);
    AllocState();
    for (h = 0; h < n; ++h) {
      x[h] = ranMT() * lx;
      y[h] = ranMT() * ly;
      th = (init == 0) ? 2.0 * PI * ranMT() : 0.0;
      vx[h] = cos(th);
      vy[h] = sin(th);
      pid[h] = h;
    }
  } else {
    FILE *fp = fopen(initfile, "rb");
    unsigned char hd[128];
    int32_t iv[6];
    size_t ok = 0;
    if (!fp) { fprintf(stderr, "nao consegui abrir %s\n", initfile); exit(1); }
    if (fread(hd, 1, 128, fp) != 128 || memcmp(hd, "VSKCHKP1", 8) != 0) {
      fprintf(stderr, "%s nao e um checkpoint VSKCHKP1\n", initfile); exit(1);
    }
    memcpy(iv, hd + 8, sizeof(iv));
    { int64_t tt; memcpy(&tt, hd + 32, 8); tchk = (long) tt; }
    n = iv[1];
    if (iv[2] != lx || iv[3] != ly) {
      fprintf(stderr, "checkpoint tem L = %d x %d, mas pedi %d x %d\n", iv[2], iv[3], lx, ly);
      exit(1);
    }
    AllocState();
    ok += fread(pid, sizeof(int32_t), n, fp);
    ok += fread(x, sizeof(double), n, fp);
    ok += fread(y, sizeof(double), n, fp);
    ok += fread(vx, sizeof(double), n, fp);
    ok += fread(vy, sizeof(double), n, fp);
    fclose(fp);
    if (ok != (size_t) 5 * n) { fprintf(stderr, "checkpoint truncado\n"); exit(1); }
    rho = (double) n / ((double) lx * ly);
  }
}

static void AllocWork(void)
{
  nbox = lx * ly;
  xtmp = xmalloc(n * sizeof(double));  ytmp = xmalloc(n * sizeof(double));
  vxtmp = xmalloc(n * sizeof(double)); vytmp = xmalloc(n * sizeof(double));
  pidtmp = xmalloc(n * sizeof(int32_t));
  box = xmalloc(n * sizeof(int));
  boxn = xmalloc(nbox * sizeof(int));
  boxs = xmalloc((nbox + 1) * sizeof(int));
  boxt = xmalloc(nbox * sizeof(int));
  vxint = xmalloc(n * sizeof(double)); vyint = xmalloc(n * sizeof(double));
  nint = xmalloc(n * sizeof(int));
  if (snap_every > 0) {
    s_th0 = xmalloc(n * sizeof(float));  s_th1 = xmalloc(n * sizeof(float));
    s_thbar = xmalloc(n * sizeof(float)); s_cloc = xmalloc(n * sizeof(float));
    s_nn = xmalloc(n * sizeof(int32_t));
  }
}

/* ------------------------------------------------------------------ um passo de Vicsek
   Estado(t) -> Estado(t+1). Se dosnap, grava (x_t, th_t, th_{t+1}, ...) com tempo tin=t.
   orderp[0] = Phi, [1] = direcao media, [2] = desvio padrao relativo da ocupacao das caixas */
static void Step(int dosnap, long tin, double *orderp)
{
  static int nnb[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {-1,0}, {1,0}, {-1,1}, {0,1}, {1,1} };
  int h, i, j, q, cn, bx, by, cx, cy, h2, nstart, nend, nstart2, nend2, hx, hy;
  double x1, y1, vx1, vy1, dx, dy, r, a, b, norm, amp, cs, sn, mean, stdv, sx, sy;

  /* caixas: counting sort */
  for (h = 0; h < nbox; ++h) boxn[h] = 0;
  for (h = 0; h < n; ++h) {
    q = ((int) x[h]) + lx * ((int) y[h]);
    ++boxn[q];
    box[h] = q;
  }
  cn = 0;
  for (h = 0; h < nbox; ++h) { boxs[h] = cn; cn += boxn[h]; boxt[h] = boxs[h]; }
  boxs[nbox] = n;

  mean = (double) n / nbox;
  stdv = 0.0;
  for (h = 0; h < nbox; ++h) { a = boxn[h] - mean; stdv += a * a; }
  stdv = sqrt(stdv / nbox) / mean;

  memcpy(xtmp, x, n * sizeof(double));   memcpy(ytmp, y, n * sizeof(double));
  memcpy(vxtmp, vx, n * sizeof(double)); memcpy(vytmp, vy, n * sizeof(double));
  memcpy(pidtmp, pid, n * sizeof(int32_t));
  for (h = 0; h < n; ++h) {
    q = boxt[box[h]]++;
    x[q] = xtmp[h]; y[q] = ytmp[h]; vx[q] = vxtmp[h]; vy[q] = vytmp[h]; pid[q] = pidtmp[h];
  }

  /* interacoes: a propria particula conta */
  for (i = 0; i < n; ++i) { nint[i] = 1; vxint[i] = vx[i]; vyint[i] = vy[i]; }

  for (h = 0; h < nbox; ++h) {
    nstart = boxs[h];
    nend = boxs[h + 1];
    hx = h % lx;
    hy = h / lx;
    for (i = nstart; i < nend; ++i) {
      x1 = x[i]; y1 = y[i]; vx1 = vx[i]; vy1 = vy[i];
      for (j = i + 1; j < nend; ++j) {                    /* mesma caixa */
        dx = x1 - x[j];
        dy = y1 - y[j];
        if (dx * dx + dy * dy < 1.0) {
          vxint[i] += vx[j]; vyint[i] += vy[j];
          vxint[j] += vx1;   vyint[j] += vy1;
          ++nint[i]; ++nint[j];
        }
      }
      for (h2 = 0; h2 < 8; ++h2) {                        /* caixas vizinhas */
        bx = hx + nnb[h2][0];
        if (bx < 0) { bx += lx; cx = -lx; } else if (bx >= lx) { bx -= lx; cx = lx; } else cx = 0;
        by = hy + nnb[h2][1];
        if (by < 0) { by += ly; cy = -ly; } else if (by >= ly) { by -= ly; cy = ly; } else cy = 0;
        q = bx + lx * by;
        if (q > h) {                                      /* cada par de caixas uma vez */
          nstart2 = boxs[q];
          nend2 = boxs[q + 1];
          for (j = nstart2; j < nend2; ++j) {
            dx = x1 - (x[j] + cx);
            dy = y1 - (y[j] + cy);
            if (dx * dx + dy * dy < 1.0) {
              vxint[i] += vx[j]; vyint[i] += vy[j];
              vxint[j] += vx1;   vyint[j] += vy1;
              ++nint[i]; ++nint[j];
            }
          }
        }
      }
    }
  }

  /* nova orientacao */
  for (i = 0; i < n; ++i) {
    dx = vxint[i];
    dy = vyint[i];
    if (dosnap) {
      s_th0[i]   = (float) atan2(vy[i], vx[i]);
      s_thbar[i] = (float) atan2(dy, dx);
      s_cloc[i]  = (float) (sqrt(dx * dx + dy * dy) / nint[i]);
      s_nn[i]    = nint[i];
    }
    if (noise == 1) {
      amp = eta * (gcnorm ? nint[i] : (1 + nint[i]));
      r = 2.0 * PI * ranMT();
      dx += amp * cos(r);
      dy += amp * sin(r);
    }
    norm = sqrt(dx * dx + dy * dy);
    if (norm > 0.0) { a = dx / norm; b = dy / norm; }
    else            { a = vx[i];     b = vy[i]; }        /* soma exatamente nula: medida zero */
    if (noise == 1) {
      vx[i] = a; vy[i] = b;
    } else {
      r = (noise == 0) ? eta * PI * (1.0 - 2.0 * ranMT()) : VonMises(kappa);
      cs = cos(r); sn = sin(r);
      vx[i] = a * cs - b * sn;
      vy[i] = b * cs + a * sn;
    }
    if (dosnap) s_th1[i] = (float) atan2(vy[i], vx[i]);
  }

  if (dosnap) WriteSnapshot(tin);                       /* x, y ainda sao x_t, y_t */

  /* parametro de ordem e deslocamento com PBC */
  sx = sy = 0.0;
  for (i = 0; i < n; ++i) {
    sx += vx[i];
    sy += vy[i];
    x[i] = wrap(x[i] + v0 * vx[i], (double) lx);
    y[i] = wrap(y[i] + v0 * vy[i], (double) ly);
  }
  sx /= n;
  sy /= n;
  orderp[0] = sqrt(sx * sx + sy * sy);
  orderp[1] = atan2(sy, sx);
  orderp[2] = stdv;
}

/* ------------------------------------------------------------------ saidas auxiliares */
static void Imaging(int s)
{
  int bn, i, j, p, h, lx2 = lx / s, ly2 = ly / s, *bb;
  bn = lx2 * ly2;
  bb = xmalloc(bn * sizeof(int));
  for (h = 0; h < bn; ++h) bb[h] = 0;
  for (h = 0; h < n; ++h) {
    p = ((int) (x[h] / s)) + lx2 * ((int) (y[h] / s));
    ++bb[p];
  }
  printf("\n");
  for (i = 0; i < ly2; ++i) {
    printf("\n");
    for (j = 0; j < lx2; ++j) printf("%d ", bb[i * lx2 + j]);
  }
  fflush(stdout);
  free(bb);
}

/* campos coarse-grained: densidade, ordem escalar local, angulo local (float32) */
static void CoarseGrainedConf(int s)
{
  static int kinit = 0, ls[2], bn, *bnn;
  static float *b0, *b1, *b2;
  static FILE *f1, *f2, *f3;
  int h, p;
  double ax, ay;
  char name[1600];
  if (!kinit) {
    kinit = 1;
    ls[0] = lx / s; ls[1] = ly / s; bn = ls[0] * ls[1];
    snprintf(name, sizeof(name), "%scg_den_%s.bin", outdir, tag); f1 = fopen(name, "wb");
    snprintf(name, sizeof(name), "%scg_sop_%s.bin", outdir, tag); f2 = fopen(name, "wb");
    snprintf(name, sizeof(name), "%scg_vop_%s.bin", outdir, tag); f3 = fopen(name, "wb");
    if (!f1 || !f2 || !f3) { fprintf(stderr, "erro abrindo arquivos cg\n"); exit(1); }
    fwrite(ls, sizeof(int), 2, f1); fwrite(ls, sizeof(int), 2, f2); fwrite(ls, sizeof(int), 2, f3);
    bnn = xmalloc(bn * sizeof(int));
    b0 = xmalloc(bn * sizeof(float)); b1 = xmalloc(bn * sizeof(float)); b2 = xmalloc(bn * sizeof(float));
  }
  if (s < 0) { if (kinit) { fclose(f1); fclose(f2); fclose(f3); } return; }   /* fechar */
  for (h = 0; h < bn; ++h) { bnn[h] = 0; b0[h] = b1[h] = b2[h] = 0.0f; }
  for (h = 0; h < n; ++h) {
    p = ((int) (x[h] / s)) + ls[0] * ((int) (y[h] / s));
    ++bnn[p];
    b1[p] += (float) vx[h];
    b2[p] += (float) vy[h];
  }
  for (p = 0; p < bn; ++p) {
    b0[p] = (float) bnn[p] / (float) (s * s);
    if (bnn[p] > 0) {
      ax = b1[p] / bnn[p];
      ay = b2[p] / bnn[p];
      b1[p] = (float) sqrt(ax * ax + ay * ay);
      b2[p] = (float) atan2(ay, ax);
    } else {
      b1[p] = 0.0f; b2[p] = 0.0f;                          /* caixa vazia: use b0 == 0 como mascara */
    }
  }
  fwrite(b0, sizeof(float), bn, f1);
  fwrite(b1, sizeof(float), bn, f2);
  fwrite(b2, sizeof(float), bn, f3);
  fflush(f1); fflush(f2); fflush(f3);
}

static void WriteMeta(const char *status, double runtime)
{
  char name[1600];
  FILE *fp;
  snprintf(name, sizeof(name), "%smeta_%s.json", outdir, tag);
  fp = fopen(name, "w");
  if (!fp) return;
  fprintf(fp, "{\n  \"status\": \"%s\",\n  \"tag\": \"%s\",\n", status, tag);
  fprintf(fp, "  \"N\": %d, \"lx\": %d, \"ly\": %d, \"rho\": %.10g, \"eta\": %.10g, \"v0\": %.10g,\n",
          n, lx, ly, rho, eta, v0);
  fprintf(fp, "  \"noise\": %d, \"gcnorm\": %d, \"kappa\": %.10g, \"seed\": %lu,\n", noise, gcnorm, kappa, seed);
  fprintf(fp, "  \"t0\": %ld, \"tmax\": %ld, \"init\": %d, \"initfile\": \"%s\", \"t_checkpoint_in\": %ld,\n", t0, tmax, init, initfile, tchk);
  fprintf(fp, "  \"ts_every\": %ld, \"snap_every\": %ld, \"snap_start\": %ld, \"cg_every\": %ld, \"cg_size\": %d,\n",
          ts_every, snap_every, snap_start, cg_every, cg_size);
  fprintf(fp, "  \"runtime_s\": %.3f\n}\n", runtime);
  fclose(fp);
}

static void Usage(const char *p)
{
  fprintf(stderr,
    "uso: %s [opcoes]\n"
    "  -L lx      lado em x (inteiro, >= 3)              [128]\n"
    "  -Y ly      lado em y (padrao: = lx)\n"
    "  -r rho     densidade                               [2.0]\n"
    "  -e eta     intensidade do ruido                    [0.30]\n"
    "  -v v0      velocidade                              [0.5]\n"
    "  -n tipo    0 angular uniforme | 1 vetorial | 2 angular von Mises  [0]\n"
    "  -G 0|1     ruido vetorial: 0 = eta*(n+1) (original), 1 = eta*n (Gregoire-Chate)  [0]\n"
    "  -k kappa   concentracao da von Mises (padrao: casa <cos> com o uniforme de mesmo eta)\n"
    "  -s seed    semente (inteiro >= 0; padrao: tempo+pid)\n"
    "  -t t0      passos de transiente sem saida          [0]\n"
    "  -T tmax    passos de producao                      [100000]\n"
    "  -i init    0 aleatorio | 1 alinhado em x | 2 checkpoint  [0]\n"
    "  -f arq     checkpoint para -i 2\n"
    "  -o dir     diretorio de saida (precisa existir)    [./]\n"
    "  -w k       grava Phi(t) a cada k passos            [1]\n"
    "  -S k       snapshot de treino a cada k passos (0 = nao)  [0]\n"
    "  -B t       primeiro passo com snapshot             [1]\n"
    "  -C k       campos coarse-grained a cada k passos (0 = nao)  [0]\n"
    "  -c s       tamanho da caixa de coarse-graining (divide lx e ly)  [1]\n"
    "  -P k       checkpoint intermediario a cada k passos (0 = so no fim)  [0]\n"
    "  -M k       matriz de densidade no stdout a cada k passos (filme)  [0]\n"
    "  -q         progresso no stderr a cada 1000 passos\n", p);
}

/* ------------------------------------------------------------------ main */
int main(int argc, char *argv[])
{
  int opt, dosnap, gz = 1;
  long t;
  double orderp[3];
  char name[1600], cmd[1800];
  FILE *fts;
  clock_t c0;
  size_t len;

  while ((opt = getopt(argc, argv, "L:Y:r:e:v:n:G:k:s:t:T:i:f:o:w:S:B:C:c:P:M:qzh")) != -1) {
    switch (opt) {
      case 'L': lx = atoi(optarg); break;
      case 'Y': ly = atoi(optarg); break;
      case 'r': rho = atof(optarg); break;
      case 'e': eta = atof(optarg); break;
      case 'v': v0 = atof(optarg); break;
      case 'n': noise = atoi(optarg); break;
      case 'G': gcnorm = atoi(optarg); break;
      case 'k': kappa = atof(optarg); break;
      case 's': seed = strtoul(optarg, NULL, 10); seed_given = 1; break;
      case 't': t0 = atol(optarg); break;
      case 'T': tmax = atol(optarg); break;
      case 'i': init = atoi(optarg); break;
      case 'f': strncpy(initfile, optarg, sizeof(initfile) - 1); break;
      case 'o': strncpy(outdir, optarg, sizeof(outdir) - 2); break;
      case 'w': ts_every = atol(optarg); break;
      case 'S': snap_every = atol(optarg); break;
      case 'B': snap_start = atol(optarg); break;
      case 'C': cg_every = atol(optarg); break;
      case 'c': cg_size = atoi(optarg); break;
      case 'P': chk_every = atol(optarg); break;
      case 'M': movie_every = atol(optarg); break;
      case 'q': verbose = 1; break;
      case 'z': gz = 0; break;                   /* nao compacta a serie temporal no fim */
      default: Usage(argv[0]); return (opt == 'h') ? 0 : 1;
    }
  }
  if (ly < 0) ly = lx;
  if (lx < 3 || ly < 3) { fprintf(stderr, "lx e ly precisam ser >= 3\n"); return 1; }
  if (noise < 0 || noise > 2) { fprintf(stderr, "-n deve ser 0, 1 ou 2\n"); return 1; }
  if (cg_every > 0 && (cg_size < 1 || lx % cg_size || ly % cg_size)) {
    fprintf(stderr, "-c deve dividir lx e ly\n"); return 1;
  }
  if (ts_every < 1) ts_every = 1;
  len = strlen(outdir);
  if (len && outdir[len - 1] != '/') { outdir[len] = '/'; outdir[len + 1] = '\0'; }
  if (!seed_given) seed = ((unsigned long) time(NULL) * 2654435761UL) ^ ((unsigned long) getpid() << 16);
  seed &= 0x7FFFFFFFUL;
  seedMT(2UL * seed + 1UL);                     /* 2s+1: sementes distintas -> sequencias distintas */
  if (noise != 2) kappa = -1.0;
  else if (kappa < 0.0) kappa = KappaFromEta(eta);

  InitialCondition();
  if (init == 2)       /* reinicio: nao repetir a sequencia aleatoria do primeiro trecho */
    seedMT(2UL * ((seed + 1000003UL * (unsigned long) (tchk + 1)) & 0x7FFFFFFFUL) + 1UL);

  snprintf(tag, sizeof(tag), "n%d%s_v%.3f_r%.3f_e%.4f_L%dx%d_s%lu",
           noise, (noise == 1) ? (gcnorm ? "G" : "o") : "", v0, rho, eta, lx, ly, seed);
  if (init == 2) {     /* reinicio nao sobrescreve as saidas do trecho anterior */
    len = strlen(tag);
    snprintf(tag + len, sizeof(tag) - len, "_c%ld", tchk);
  }
  AllocWork();
  WriteMeta("running", 0.0);

  snprintf(name, sizeof(name), "%sts_%s.dat", outdir, tag);
  fts = fopen(name, "w");
  if (!fts) { fprintf(stderr, "nao consegui abrir %s (o diretorio existe?)\n", name); return 1; }
  fprintf(fts, "# t\tPhi\tdirecao\trsd_caixas   | N=%d lx=%d ly=%d rho=%g eta=%g v0=%g noise=%d gc=%d kappa=%g seed=%lu t0=%ld\n",
          n, lx, ly, rho, eta, v0, noise, gcnorm, kappa, seed, t0);
  if (snap_every > 0) {
    snprintf(name, sizeof(name), "%ssnap_%s.bin", outdir, tag);
    fsnap = fopen(name, "wb");
    if (!fsnap) { fprintf(stderr, "nao consegui abrir %s\n", name); return 1; }
  }

  c0 = clock();
  for (t = 1; t <= t0; ++t) {
    Step(0, 0, orderp);
    if (verbose && t % 1000 == 0) fprintf(stderr, "transiente t=%ld Phi=%.4f\n", t, orderp[0]);
  }
  if (cg_every > 0) CoarseGrainedConf(cg_size);             /* t = 0 de producao */

  for (t = 1; t <= tmax; ++t) {
    dosnap = (snap_every > 0 && t >= snap_start && (t % snap_every) == 0);
    Step(dosnap, t - 1, orderp);                            /* estado t-1 -> t */
    if (t % ts_every == 0) fprintf(fts, "%ld\t%.8e\t%.8e\t%.8e\n", t, orderp[0], orderp[1], orderp[2]);
    if (verbose && t % 1000 == 0) fprintf(stderr, "t=%ld Phi=%.4f\n", t, orderp[0]);
    if (cg_every > 0 && t % cg_every == 0) CoarseGrainedConf(cg_size);
    if (movie_every > 0 && t % movie_every == 0) Imaging(1);
    if (chk_every > 0 && t % chk_every == 0) WriteCheckpoint(tchk + t0 + t);
  }
  WriteCheckpoint(tchk + t0 + tmax);   /* tempo acumulado desde o inicio da cadeia de reinicios */
  fclose(fts);
  if (fsnap) fclose(fsnap);
  if (cg_every > 0) CoarseGrainedConf(-1);
  WriteMeta("done", (double) (clock() - c0) / CLOCKS_PER_SEC);
  if (verbose) fprintf(stderr, "tempo de CPU: %.2f s\n", (double) (clock() - c0) / CLOCKS_PER_SEC);

  if (gz) {
    snprintf(cmd, sizeof(cmd), "gzip -f %sts_%s.dat", outdir, tag);
    if (system(cmd) != 0) fprintf(stderr, "aviso: gzip falhou\n");
  }
  return 0;
}
