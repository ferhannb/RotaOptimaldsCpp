# RotaOptimaldsPy

`RotaOptimaldsCpp/` projesinin Python/CasADi portudur. Aynı senaryo dosyasi formatini, ayni receding-horizon MPC akisini ve ayni dairesel obstacle detour mantigini ayri bir klasorde sunar.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Calistirma

Varsayilan senaryo:

```bash
python3 main.py
```

Belirli senaryo:

```bash
python3 main.py --scenario scenarios/rotaoptimalds_default.ini
```

CSV sonuclarini cizdirmek icin:

```bash
python3 plot_receding.py --log receding_log.csv --wp waypoints.csv --scenario scenarios/rotaoptimalds_default.ini
```

## Icerik

- `main.py`: CLI giris noktasi
- `rota_optimal_ds.py`: veri modelleri, CasADi MPC modeli ve receding-horizon dongusu
- `scenario_parser.py`: `.ini` senaryo parser'i
- `obstacle_avoidance.py`: obstacle tetikleme ve detour waypoint secimi
- `plot_receding.py`: CSV cizim araci
- `scenarios/`: ornek senaryolar

Not: Bu port, C++ uygulamasinin davranisini olabildigince dogrudan tasir. Bu nedenle bazi solver ayrintilari ve warm-start tercihleri de bilerek benzer tutulmustur.
