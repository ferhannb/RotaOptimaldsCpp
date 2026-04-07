# RotaOptimaldsCpp

`Fresnel/RotaOptimalds.py` akışının C++/CasADi sürümü.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

`CASADI_ROOT` vererek CasADi kök klasörünü belirleyebilirsin:

```bash
cmake -S . -B build -DCASADI_ROOT=/path/to/casadi
```

## Run

```bash
./build/rota_optimal_ds
```

Çalışma sonunda `receding_log.csv` ve `waypoints.csv` üretilir.

CLI ile senaryo dosyası vererek:

```bash
./build/rota_optimal_ds \
  --scenario scenarios/rotaoptimalds_default.ini \
  --out-log receding_log.csv \
  --out-wp waypoints.csv
```

Yardım:

```bash
./build/rota_optimal_ds --help
```

> Not: Sisteminizde `ipopt` plugin'i yoksa kod otomatik `sqpmethod` fallback kullanır.
> Python ile birebir davranış için `ipopt` kurulumu önerilir.

## Windows Hazırlık

### 1) Gerekenler

- Visual Studio 2022 (Desktop development with C++)
- CMake (>=3.16)
- CasADi C++ dağıtımı (içinde `include/`, `lib/` veya `bin/` olmalı)

Önerilen klasör yapısı:

```text
RotaOptimaldsCpp/
  third_party/
    casadi/
      include/
      lib/
      bin/
```

### 2) Build (PowerShell)

```powershell
cd C:\path\to\RotaOptimaldsCpp
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCASADI_ROOT="$PWD\\third_party\\casadi"
cmake --build build --config Release
```

### 3) Run

```powershell
.\build\Release\rota_optimal_ds.exe --scenario .\scenarios\rotaoptimalds_default.ini
```

`ROTA_COPY_CASADI_RUNTIME=ON` (default) ise CMake, CasADi ile ilgili DLL'leri exe yanına kopyalamaya çalışır.

### 4) Dağıtım İçin Tek Klasör

```powershell
cmake --install build --config Release --prefix .\dist
```

`dist` klasörünü zipleyip diğer Windows makineye taşıyabilirsin.
Gerekirse hedef makinede `vcruntime` eksikse Visual C++ Redistributable kur.

## Plot

```bash
python3 plot_receding.py --log receding_log.csv --wp waypoints.csv --scenario scenarios/rotaoptimalds_obstacle.ini
```

## Scenario Dosyasi

- Varsayılan örnek: `scenarios/rotaoptimalds_default.ini`
- `waypoint` satırı formatı:
  `waypoint = X,Y,psig,Kf,tol,use_Kf,w_wp,hit_scale`
- `obstacle` satırı formatı:
  `obstacle = cx,cy,radius[,enabled]`
- Birden fazla engel icin `obstacle` satırını tekrar edebilirsin.
- Toplu yükleme icin:
  `obstacles_csv = scenarios/obstacles_many.csv`
  CSV formatı: `cx,cy,radius[,enabled]` (header opsiyonel)
- Engel kacınma ayarları:
  `enable_obstacle_avoidance`, `obstacle_clearance`, `obstacle_trigger_margin`, `obstacle_waypoint_tol`
- Boş/opsiyonel alanlar için `none` veya boş değer kullanabilirsin.
- Örnek engel senaryosu: `scenarios/rotaoptimalds_obstacle.ini`

## İçerik

- `src/rota_optimal_ds.hpp/.cpp`
  - MPC modeli (Opti + IPOPT)
  - clothoid dynamics + fixed-ramp `K` güncellemesi
  - multi-waypoint receding-horizon döngüsü
  - warm-start/solution shift mantığı
- `src/main.cpp`
  - CLI + scenario dosyasından çalışma
- `src/scenario_parser.hpp/.cpp`
  - ini tarzı scenario parser
- `plot_receding.py`
  - C++ çıktısından Python'daki 2x2 grafik setini üretir
# RotaOptimaldsCpp
# RotaOptimaldsCpp
