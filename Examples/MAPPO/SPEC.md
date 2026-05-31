# CleanRL MAPPO 実装仕様書

**プロジェクト**: B-ACE-MARL  
**ブランチ**: `Add-MARL-Example-with-CleanRL`  
**作成日**: 2026-05-31  
**対象アルゴリズム**: MAPPO (Multi-Agent Proximal Policy Optimization)  
**学習パラダイム**: CTDE (Centralized Training, Decentralized Execution)

---

## 1. 概要

### 1.1 目的

CleanRL の「シンプル・単一責任・読みやすいコード」という哲学に基づき、B-ACE 空戦シミュレーション環境向けの MAPPO 実装を追加する。既存の BenchMARL・Tianshou 実装と同等の機能をより軽量な依存関係で実現する。

### 1.2 前提条件

| 項目 | 内容 |
|------|------|
| 環境 | B-ACE Godot シミュレーター (`B_ACE_GodotPettingZooWrapper` 経由) |
| 行動空間 | 連続値 4 次元 `[hdg, level, g_force, fire]` |
| 観測空間 | 連続値ベクトル (構成するエージェント数に依存、下記参照) |
| Blue エージェント | RL 制御 (`base_behavior: "external"`, DCA ミッション) |
| Red エージェント | スクリプト制御 (`base_behavior: "baseline1"`) |

---

## 2. 環境の空間定義

### 2.1 観測空間 (State Space)

各 Blue エージェントは以下の成分からなるフラット NumPy 配列を観測として受け取る (Readme §RL Spaces Definition より)。

| 成分 | 次元数 | 内容 |
|------|--------|------|
| Agent State | **8** | 自機情報 (位置 3, 針路+速度 2, 目標距離+アスペクト角 2, ミサイル残弾 1) |
| Allies Info (味方 1 機ごと) | **6** | 高度差+距離 2, アスペクト角+オフアングル 2, 目標距離 1, 更新フラグ 1 |
| Enemies Info (敵 1 機ごと) | **11** | 高度差+距離 2, アスペクト角+オフアングル 2, 目標距離 1, 自→敵 WEZ 2, 敵→自 WEZ 2, 被照準フラグ 1, 更新フラグ 1 |

**観測次元の計算式:**

```
obs_dim = 8 + 6 × n_allies + 11 × n_enemies
```

**代表的なシナリオでの obs_dim:**

| シナリオ | n_blue (= n_allies+1) | n_red (= n_enemies) | obs_dim |
|---------|----------------------|---------------------|---------|
| 1v1 | 1 (味方 0) | 1 | 8 + 0 + 11 = **19** |
| 2v2 | 2 (味方 1) | 2 | 8 + 6 + 22 = **36** |
| 4v4 | 4 (味方 3) | 4 | 8 + 18 + 44 = **70** |

> **注意**: Centralized Critic の入力は `global_obs_dim = obs_dim × n_blue_agents` となる。

### 2.2 行動空間 (Action Space)

行動は 4 次元の連続値配列 `[hdg, level, g_force, fire]` (`run_simple_example.py` の定義に準拠)。

| インデックス | 名称 | 範囲 | 意味 |
|------------|------|------|------|
| 0 | `hdg` | `[-1.0, 1.0]` | 希望針路変化量 (`値 × 180°` が右方向への変化量) |
| 1 | `level` | `[-1.0, 1.0]` | 希望飛行高度 (`値 × 25,000ft + 25,000ft`) |
| 2 | `g_force` | `[-1.0, 1.0]` | 希望 G 荷重 (`-1.0` → 1G、`+1.0` → max G) |
| 3 | `fire` | `[-1.0, 1.0]` | `> 0.0` でミサイル発射、`≤ 0.0` で非発射 |

> **実装上のポイント**: Actor の出力は Tanh で `[-1, 1]` に正規化する。`fire` 次元はネットワーク出力をそのまま渡し、シミュレーター側でしきい値 0 により二値判定される。

**サンプル行動 (`run_simple_example.py` より):**
```python
actions[agent] = [0.1 * turn_side, 0.5, -0.75, 0.0]
# hdg: 右 or 左 18° 変化, level: 37,500ft, g_force: 低G旋回, fire: 発射しない
```

---

## 3. ディレクトリ構成

```
Examples/
└── CleanRL/
    ├── SPEC.md                 # 本仕様書
    ├── config.yaml             # ハイパーパラメータ・環境設定
    ├── mappo_b_ace.py          # エントリーポイント (学習ループ)
    ├── networks.py             # Actor / Centralized Critic ネットワーク定義
    ├── vec_env.py              # PettingZoo → 並列 VecEnv 変換ユーティリティ
    └── Readme.md
```

---

## 4. アルゴリズム仕様 — MAPPO with CTDE

### 4.1 基本方針

```
┌─────────────────────────────────────────────────────────────────┐
│                     Centralized Critic                          │
│  入力: 全 Blue エージェント観測を結合した global_obs             │
│         global_obs_dim = obs_dim × n_blue_agents               │
│  出力: V(s_global)  — shape (batch, 1)                         │
└─────────────────────────────────────────────────────────────────┘
               ↑ 訓練時のみ使用 (CTDE の C 部)

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Actor 0    │  │   Actor 1    │  │     ...      │
│  π(a | o₀)  │  │  π(a | o₁)  │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
  ↑ 実行時は各エージェントの局所観測のみ使用 (CTDE の DE 部)
  ↑ share_policy_params=True の場合、全 Blue が同一ネットワーク重みを共有
```

### 4.2 アルゴリズムフロー

```
for iteration in range(num_iterations):

    # ─── Rollout フェーズ (経験収集) ───────────────────────────
    for step in range(num_steps):
        obs[t]    ← 各エージェントの局所観測  shape: (n_envs, n_agents, obs_dim)
        actions[t], logprobs[t] ← Actor(obs[t])     # 分散実行
        values[t]               ← Critic(global_obs[t])
        obs[t+1], rewards[t], dones[t] ← env.step(actions[t])

    # ─── Advantage 計算 (GAE) ──────────────────────────────────
    advantages ← GAE(rewards, values, dones, gamma, gae_lambda)
    returns    ← advantages + values

    # ─── Policy Update フェーズ ────────────────────────────────
    for epoch in range(update_epochs):
        for minibatch in shuffle(rollout_buffer):
            # --- Actor 損失 (PPO clip) ---
            ratio     = exp(logprob_new - logprob_old)
            clip_loss = -mean( min(ratio × adv,  clip(ratio, 1−ε, 1+ε) × adv) )

            # --- Critic 損失 (optional value clip) ---
            value_loss = MSE(Critic(global_obs), returns)

            # --- Entropy ボーナス ---
            entropy_loss = -mean(entropy(π))

            total_loss = clip_loss + vf_coef × value_loss + ent_coef × entropy_loss
            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
```

### 4.3 Centralized Critic の global_obs 構成

```python
# 例: 2 Blue エージェント, obs_dim=36 → global_obs_dim=72
global_obs = torch.cat([obs[:, 0, :], obs[:, 1, :]], dim=-1)  # (batch, global_obs_dim)
value = critic(global_obs)                                      # (batch, 1)
```

---

## 5. モジュール仕様

### 5.1 `networks.py`

#### `MAPPOActor`

| 項目 | 内容 |
|------|------|
| 入力 | 局所観測 `obs` — shape `(batch, obs_dim)` |
| 出力 | 行動分布の平均 `mu`、学習可能な `log_std` (parameter) |
| アーキテクチャ | MLP: `obs_dim → hidden[0] → hidden[1] → action_dim` |
| 中間活性化 | Tanh |
| 出力活性化 | なし (mu はそのまま、Tanh は `get_action` 内でサンプル後に適用) |
| 初期化 | Orthogonal (gain=√2)、最終層のみ gain=0.01 |

```python
class MAPPOActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]): ...
    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        # returns (mu, log_std)   ← log_std is nn.Parameter, not from forward
    def get_action(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # returns (action, log_prob, entropy)
        # action はサンプル後に Tanh で [-1, 1] にクランプ
```

#### `MAPPOCritic` (Centralized)

| 項目 | 内容 |
|------|------|
| 入力 | `global_obs` — shape `(batch, obs_dim × n_blue_agents)` |
| 出力 | 状態価値 `V(s)` — shape `(batch, 1)` |
| アーキテクチャ | MLP: `global_obs_dim → hidden[0] → hidden[1] → 1` |
| 中間活性化 | Tanh |
| 初期化 | Orthogonal (gain=√2)、最終層のみ gain=1.0 |

```python
class MAPPOCritic(nn.Module):
    def __init__(self, global_obs_dim: int, hidden_sizes: list[int]): ...
    def forward(self, global_obs: Tensor) -> Tensor:
        # returns value: (batch, 1)
```

---

### 5.2 `vec_env.py`

PettingZoo Parallel API の `B_ACE_GodotPettingZooWrapper` を、複数環境を並列実行できる形式に変換する **自前実装** ユーティリティ (外部ライブラリ supersuit は使用しない)。

#### `make_env(config: dict, env_idx: int, random_seed: int) -> Callable`

- `B_ACE_GodotPettingZooWrapper` のファクトリ関数を返す
- `port = config.env.port + env_idx` でポート競合を回避
- 各環境のランダムシード (乱数初期化用の数値) を `random_seed + env_idx` で個別設定

#### `PettingZooVecEnv`

`n_envs` 個の B-ACE プロセスをまとめて操作するクラス。

| メソッド | 入出力 |
|----------|-------|
| `reset()` | → `obs`: shape `(n_envs, n_agents, obs_dim)` |
| `step(actions)` | `actions`: `(n_envs, n_agents, action_dim)` → `obs, rewards, dones, infos` |
| `get_global_obs(obs)` | `obs (n_envs, n_agents, obs_dim)` → `global_obs (n_envs, global_obs_dim)` |
| `close()` | 全 Godot プロセスをシャットダウン |

> **並列実行戦略**: 各環境は独立した Godot プロセスとして起動されるため、`subprocess` レベルの並列化になる。Python のスレッド/プロセス並列 (`multiprocessing`) は使わず、`step` をラウンドロビンで順次呼び出す方式 (DummyVectorEnv 相当) を基本とする。

---

### 5.3 `mappo_b_ace.py`

エントリーポイント。以下の処理を順に実行する。

```
1. config.yaml 読み込み → CLI 引数でオーバーライド (argparse)
2. TensorBoard SummaryWriter 初期化
3. ランダムシード (乱数初期化用の数値) を torch / numpy に設定
4. PettingZooVecEnv 初期化 (n_envs 並列)
5. obs_dim / action_dim を環境から自動取得
6. MAPPOActor, MAPPOCritic をインスタンス化
7. Adam オプティマイザ設定 (Actor と Critic を単一 optimizer で管理)
8. 学習ループ (num_iterations = total_timesteps // (num_envs × num_steps) 回)
   a. Rollout 収集 → バッファ (obs, actions, logprobs, rewards, dones, values) に格納
   b. GAE でアドバンテージと returns を計算
   c. update_epochs × num_minibatches 回のミニバッチ更新
   d. TensorBoard へのログ記録
   e. checkpoint_interval ごとにチェックポイント保存
9. 最終ポリシー保存 (actor.pt, critic.pt)
```

#### ログ項目 (TensorBoard)

| キー | 内容 |
|------|------|
| `charts/episodic_return` | エピソード累積報酬 (全エージェント合計) |
| `charts/episodic_length` | エピソード長 (ステップ数) |
| `charts/SPS` | 1 秒あたりの環境ステップ数 |
| `losses/policy_loss` | Actor の PPO clip 損失 |
| `losses/value_loss` | Critic の MSE 損失 |
| `losses/entropy` | 行動エントロピー |
| `losses/approx_kl` | 近似 KL ダイバージェンス |
| `losses/clipfrac` | クリップされた更新の割合 |
| `losses/explained_variance` | 価値関数の説明分散 |

---

## 6. 設定ファイル仕様 (`config.yaml`)

```yaml
# ========== 環境設定 ==========
env:
  env_path: "../../bin/B_ACE_v0.1.exe"
  port: 12500           # 並列環境は 12500, 12501, ... と自動オフセット
  renderize: 0          # 0: ヘッドレス (訓練用), 1: 表示あり (デバッグ用)
  speed_up: 50000
  max_cycles: 36000
  action_type: "Low_Level_Continuous"

# ========== エージェント設定 ==========
agents:
  blue:
    num_agents: 1             # Blue (RL制御) エージェント数
    base_behavior: "external" # "external" = RL エージェントとして動作
    mission: "DCA"
    init_position: {x: 0.0, y: 25000.0, z: 30.0}
    init_hdg: 0.0
    target_position: {x: 0.0, y: 25000.0, z: 30.0}
    rnd_offset_range: {x: 10.0, y: 10000.0, z: 5.0}
  red:
    num_agents: 1             # Red (スクリプト制御) エージェント数
    base_behavior: "baseline1"
    mission: "striker"
    init_position: {x: 0.0, y: 25000.0, z: -30.0}
    init_hdg: 180.0
    beh_config:
      dShot:  [0.50, 0.99, 1.04]
      lCrank: [0.98, 0.96, 1.14]
      lBreak: [1.17, 0.51, 1.05]

# ========== 報酬設定 ==========
rewards:
  mission_factor:              0.001
  missile_fire_factor:        -0.1
  missile_no_fire_factor:     -0.001
  missile_miss_factor:        -0.5
  detect_loss_factor:         -0.1
  keep_track_factor:           0.001
  hit_enemy_factor:            3.0
  hit_own_factor:             -5.0
  mission_accomplished_factor: 10.0

# ========== アルゴリズム設定 ==========
algo:
  # --- 訓練環境 ---
  num_envs: 4           # 並列起動する訓練環境 (Godotプロセス) の数
  random_seed: 42       # 乱数初期化の種 (再現性のための固定値)

  # --- ロールアウト ---
  num_steps: 512        # 1 訓練環境あたりの 1 イテレーション収集ステップ数
  total_timesteps: 3000000

  # --- PPO ハイパーパラメータ ---
  update_epochs: 10
  num_minibatches: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  clip_vloss: true
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  target_kl: null       # null で KL 早期停止なし

  # --- 学習率 ---
  learning_rate: 3.0e-4
  anneal_lr: true       # total_timesteps にわたって線形 decay

  # --- ネットワーク ---
  hidden_sizes: [256, 256]
  share_policy_params: true   # Blue 全エージェントでネットワーク重みを共有

# ========== ログ・保存 ==========
logging:
  exp_name: "mappo_b_ace"
  save_dir: "Results"
  checkpoint_interval: 100    # イテレーション単位でチェックポイント保存
```

---

## 7. 実行インターフェース

### 7.1 基本実行

```bash
python Examples/CleanRL/mappo_b_ace.py
```

### 7.2 CLI オーバーライド

```bash
# 2v2 シナリオで並列 8 環境、500 万ステップ訓練
python Examples/CleanRL/mappo_b_ace.py \
  --agents.blue.num_agents 2 \
  --agents.red.num_agents  2 \
  --algo.num_envs 8 \
  --algo.total_timesteps 5000000 \
  --logging.exp_name "mappo_2v2"

# 学習率を変更
python Examples/CleanRL/mappo_b_ace.py --algo.learning_rate 1e-4
```

### 7.3 チェックポイントから再開

```bash
python Examples/CleanRL/mappo_b_ace.py \
  --restore Results/mappo_b_ace_20260531/checkpoint_100.pt
```

---

## 8. 依存パッケージ

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| `torch` | ≥ 2.0 | ニューラルネットワーク・最適化 |
| `numpy` | ≥ 1.24 | 配列操作 |
| `pettingzoo` | ≥ 1.24 | マルチエージェント環境 API |
| `tensorboard` | ≥ 2.14 | 学習ログ可視化 |
| `gymnasium` | ≥ 0.29 | Gym 互換 API |
| `pyyaml` | ≥ 6.0 | 設定ファイル読み込み |

> SuperSuit は使用しない。VectorEnv は `vec_env.py` で自前実装する。

---

## 9. 実装上の注意事項

### 9.1 ポート管理

並列訓練環境の起動時、各 Godot プロセスは別々の TCP ポートを使用する。

```python
port = config.env.port + env_idx  # 12500, 12501, 12502, ...
```

### 9.2 obs_dim の動的計算

`num_agents` の変更時に obs_dim と global_obs_dim が自動で変わるよう、環境オブジェクトから取得する。

```python
obs_dim        = env.observation_space(env.agents[0]).shape[0]
global_obs_dim = obs_dim * n_blue_agents
```

### 9.3 終了フラグの統一

PettingZoo の `terminations` / `truncations` は全エージェント分の辞書。`done` フラグは論理和で統一する。

```python
done = any(terminations.values()) or any(truncations.values())
```

### 9.4 チェックポイント保存内容

```python
{
    "actor_state_dict":     actor.state_dict(),
    "critic_state_dict":    critic.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "iteration":            current_iteration,
    "global_step":          global_step,
    "config":               config,
}
```

---

## 10. 用語整理

| 用語 | 本仕様書での意味 |
|------|----------------|
| **ランダムシード** (`random_seed`) | 乱数生成器を初期化する整数値。同じ値を指定すると再現性のある結果が得られる |
| **訓練環境** (`num_envs`) | ロールアウト (経験収集) のために並列起動する Godot プロセスの数。シードとは別概念 |
| **実験シード** | BenchMARL 同様に「異なるランダムシードで同じ設定を複数回訓練して統計的信頼性を確認する」用途。本実装では CLI で `--algo.random_seed` を変えて手動実行 |

---

## 11. 既存実装との対応表

| 機能 | BenchMARL | Tianshou | **MAPPO (本実装)** |
|------|-----------|----------|---------------------|
| アルゴリズム | ISAC / MAPPO 等 | DDPG | **MAPPO** |
| Critic | 各実装依存 | 独立 (IDDPG) | **集中型 (CTDE)** |
| 環境ラッパー | BenchMARL 専用 | PettingZoo | **PettingZoo** |
| ログ | TensorBoard | なし | **TensorBoard** |
| 設定 | YAML | Python dict | **YAML のみ** |
| 並列化 | TorchRL | DummyVectorEnv | **ThreadPoolExecutor** |
| 主要依存 | BenchMARL, TorchRL | Tianshou | **PyTorch のみ** |

---

## 12. 保留事項

### 12.1 CPU 利用率の向上（並列化改善）

**概要**: 現在の `ThreadPoolExecutor` による並列化は TCP ソケット I/O の並列化に効果的だが、Python の学習ループ（GAE 計算・PPO 更新）は単一コアのみ使用している。

**やりたいこと**: 学習ループ自体を GPU または複数コアで高速化する。

**検討内容**:
- PyTorch の GPU 対応（`device: cuda`）
- 複数ワーカーによる rollout 収集の並列化

**現状**: CPU 使用率 50%、メモリ使用率 16% の環境で `num_envs=28` を使用中。

---

### 12.2 戦闘空域の Godot 可視化

**概要**: `config.yaml` の `combat_area` で定義した戦闘空域の境界を、Godot のシミュレーション画面上に描画する。

**やりたいこと**:
- `renderize: 1` のとき、戦闘空域の四隅を赤線の矩形で表示する
- 空域外に出た機体を視覚的に識別しやすくする

**実装方針**:
1. `ViewPort.gd` に `draw_combat_area(mesh, area)` 関数を追加（`draw_grid()` と同じ `ImmediateMesh` + `PRIMITIVE_LINES` を使用）
2. `build_b_ace_config()` で `combat_area` を Godot 世界座標に変換して `EnvConfig` に追加
3. `SimManager.gd` または `B_ACE.gd` で config を受け取り Viewport に渡す

**必要なもの**:
- Godot Engine 4.4（無料）
- .NET SDK 6.0 以上（C# コンポーネント用）
- エクスポートテンプレート（.exe 再生成時のみ）

**未解決の課題**:
- 正規化 obs 座標（`own_x_pos`, `own_z_pos`）と Godot 世界座標のスケール変換係数が未確定
- `SimManager.gd` の `initialize()` にデバッグ出力を追加して Blue 機の初期 `global_position` を確認することで解決できる

**開発手順**（確認済み）:
```yaml
# config.yaml でデバッグモードにする
env:
  env_path: "debug"   # Godot エディタから ▶ Play して接続
```
→ Godot エディタから実行すれば .exe の再エクスポートなしにテスト可能
