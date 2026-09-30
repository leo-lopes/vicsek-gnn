# Site e vídeo

O site do projeto fica em `../docs/` (GitHub Pages). Esta pasta tem os scripts do vídeo "rede que nunca viu bandas × modelo exato".

- `ptload.py`: lê os `modelo.pt` sem PyTorch (zip + pickle).
- `m4np.py`: o M4 e a cabeça "hist" do `treino.py` em NumPy, mais um passo de malha fechada (rede ou Vicsek exato).
- `validar_m4.py`: confere a reimplementação no conjunto de teste do ruído vetorial. Resultado: gap e erro de direção iguais aos do `treino.py` (GL: 0,2059 e 0,97° no gás; 0,1966 e 0,67° nas bandas).
- `gerar_video.py`: `python3 gerar_video.py sim` roda as duas simulações (caixa 192 × 48, ρ = 0,5, η = 0,55, mesma condição inicial) e `python3 gerar_video.py render` grava o MP4 com o ffmpeg.

Checagem em malha fechada (caixa 64 × 64, 3 mil + 7 mil passos): rede GL com ρ = 0,5 dá Φ = 0,597 ± 0,007 (rollout.py: 0,600).

Rode da raiz do repositório, com numpy, scipy e matplotlib.
