#!/usr/bin/env bash
# Собрать python-3.11 окружение для industrial_ad на NFS (опционально).
# Адаптация проверенного bootstrap_env.sh из Learn2Design: та же лестница
# стратегий (conda -> venv -> virtualenv -> pip --target), тот же env-json
# контракт — phase0_repro.py читает ad_env.json и сам exec-ается в нужный python.
#
# Запускать ОДИН раз из терминала Jupyter Server (у него есть сеть):
#   bash scripts/phase0/bootstrap_ad_env.sh
#
# Если решено работать на python 3.10 из базового образа (для industrial_ad
# этого достаточно) — этот скрипт НЕ нужен; достаточно:
#   pip install --user pywavelets pandas scikit-learn scipy pyarrow tqdm
set -uo pipefail

NFS=${NFS:-/home/jovyan/kovalenko}
VENV=$NFS/venvs/ad311
TARGET=$NFS/venvs/ad311-site
REPO=${REPO:-$NFS/industrial_ad-main}
ENVJSON=$NFS/venvs/ad_env.json
mkdir -p "$NFS/venvs" /home/jovyan/tmp

say() { printf '\n== %s ==\n' "$*"; }

say "ищу python >= 3.11"
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  command -v "$c" >/dev/null 2>&1 || continue
  "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,11) else 1)' 2>/dev/null || continue
  PY=$(command -v "$c"); break
done
[ -n "$PY" ] || { echo "BLOCKER: нет python 3.11+ на PATH (в L2D находился)."; exit 1; }
echo "using $PY ($("$PY" -V 2>&1))"

MODE=""; ENVPY=""; ENVPP=""

if command -v conda >/dev/null 2>&1; then
  say "strategy conda"
  if conda create -y -p "$VENV" python=3.11 pip >/dev/null 2>&1; then
    MODE=conda; ENVPY="$VENV/bin/python"; echo "created $VENV"
  else
    echo "conda create failed, falling through"
  fi
fi

if [ -z "$MODE" ]; then
  say "strategy venv"
  rm -rf "$VENV"
  if "$PY" -m venv "$VENV" >/dev/null 2>&1 && [ -x "$VENV/bin/python" ]; then
    MODE=venv; ENVPY="$VENV/bin/python"; echo "created $VENV"
  else
    echo "python -m venv failed (Debian: ensurepip отдельным пакетом) — falling through"
    rm -rf "$VENV"
  fi
fi

if [ -z "$MODE" ]; then
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    say "bootstrapping pip (get-pip.py)"
    curl -fsSL https://bootstrap.pypa.io/pip/get-pip.py -o /tmp/get-pip.py \
      && "$PY" /tmp/get-pip.py --user >/dev/null 2>&1 && echo "pip ok" || echo "get-pip failed"
  fi
  say "strategy virtualenv"
  "$PY" -m pip install -q --user virtualenv 2>/dev/null
  rm -rf "$VENV"
  if "$PY" -m virtualenv -q "$VENV" >/dev/null 2>&1 && [ -x "$VENV/bin/python" ]; then
    MODE=virtualenv; ENVPY="$VENV/bin/python"; echo "created $VENV"
  else
    rm -rf "$VENV"
  fi
fi

if [ -z "$MODE" ]; then
  say "strategy target (PYTHONPATH, без venv)"
  mkdir -p "$TARGET"
  MODE=target; ENVPY="$PY"; ENVPP="$TARGET"
fi

say "disk check (torch cu12 тянет ~2.5-3 ГБ wheels; L2D предупреждал: квота NFS тесная)"
df -h /home/jovyan | tail -1
AVAIL_KB=$(df -Pk /home/jovyan | tail -1 | awk '{print $4}')
echo "free: $((AVAIL_KB / 1024 / 1024)) GB (нужно >= 6)"
[ $((AVAIL_KB / 1024 / 1024)) -ge 6 ] || { echo "BLOCKER: мало места — почистить NFS или работать на py310 образа"; exit 1; }

say "install torch + deps (mode: $MODE) — долгий шаг, 10-30 мин на NFS это норма"
if [ "$MODE" = target ]; then
  PIP=("$PY" -m pip install --target="$TARGET" --upgrade --progress-bar off --no-input)
else
  "$ENVPY" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$ENVPY" -m pip install -q --upgrade pip wheel setuptools
  PIP=("$ENVPY" -m pip install --upgrade --progress-bar off --no-input)
fi
# пин по умолчанию = версия torch в JS-образе воркспейса (2.6.0+cu124, факт 2026-08-13);
# переопределить: TORCH_SPEC="torch==2.7.0" bash bootstrap_ad_env.sh
TORCH_SPEC=${TORCH_SPEC:-"torch==2.6.0"}
"${PIP[@]}" "$TORCH_SPEC" 2>&1 | tail -3 || { echo "FAILED installing $TORCH_SPEC"; exit 1; }
"${PIP[@]}" -r "$REPO/requirements.txt" 2>&1 | tail -3 || { echo "FAILED installing requirements"; exit 1; }

say "записываю контракт для раннера"
cat > "$ENVJSON" <<JSON
{"mode": "$MODE", "python": "$ENVPY", "pythonpath": "$ENVPP"}
JSON
cat "$ENVJSON"

say "verify (CPU здесь ок; GPU проверит первый job)"
env PYTHONPATH="${ENVPP}${PYTHONPATH:+:$PYTHONPATH}" "$ENVPY" - <<PY
import sys; print("python", sys.version.split()[0])
import torch; print("torch", torch.__version__, "cuda build:", torch.version.cuda)
sys.path.insert(0, "$REPO")
import industrial_ad
print("industrial_ad import OK")
PY
[ $? -eq 0 ] && echo "OK — дальше: python3.11 scripts/phase0/phase0_repro.py --list-only" || { echo "verification FAILED"; exit 1; }
