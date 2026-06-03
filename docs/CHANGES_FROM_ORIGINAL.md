# オリジナルからの変更点まとめ

> 比較対象: [andrekuros/B-ACE](https://github.com/andrekuros/B-ACE) (`main` ブランチ)  
> 本ブランチ: `add-interface-to-change-asset-specification`  
> 変更コミット数: 6 件（`988d9f7` 〜 `e7f0f13`）

---

## 1. 新規ファイル

| ファイル | 説明 |
|---|---|
| `b_ace_py/red_team_policy.py` | Pythonサイドのルールベース Red team ポリシー |
| `Examples/MAPPO/fighter_spec.json` | カスタム機体スペック（max_g=24.6 で最小旋回半径 0.25 NM） |
| `Examples/MAPPO/missile_spec.json` | カスタムミサイルスペック |
| `bin/B_ACE_v0.1_custom.exe` | Godot ソース変更を反映したカスタムビルドバイナリ |
| `docs/CHANGES_FROM_ORIGINAL.md` | 本ファイル |
| `docs/RL_DATA_FLOW.md` | 強化学習データフロー解説 |

---

## 2. カテゴリ別変更詳細

### 2-1. External Red Team ポリシーシステム（最大変更）

オリジナルでは Red team は常に Godot 内の FSM（`baseline1` 等）で制御されていた。  
本ブランチでは Red team を Python から制御できる **External Policy** アーキテクチャを追加。

#### 2-1-1. `b_ace_py/red_team_policy.py`（新規）

`RedTeamPolicy` クラス。Red agent の観測から行動を生成するルールベースモデル。

```
主要メソッド:
  act(agent_name, obs)          1エージェント分の行動を返す
  act_batch(agent_names, obs)   複数エージェントの行動を一括生成
  _best_target(obs, labels)     alive トラックの中で最優先ターゲットを選択
  _boundary_heading(obs, labels) combat_area 境界回避ヘッディングを計算
```

**ポリシー内容:**
- `track_dist_{id} >= 0`（alive）なら全方向から aspect_angle でチェイス  
  （検出圏外（±60° FOV 外）でも geometrically 正確な bearing を利用）
- `track_detected_{id} == 1` の時のみミサイル発射（確率的、`shot_threshold ± shot_variation`）
- `combat_area` 境界から `boundary_margin_nm` 以内に入ったら中心方向へ強制旋回（`strength` でブレンド）

#### 2-1-2. `Godot_Air_Combat/SimManager.gd`

```gdscript
# 追加変数
var red_external_agents = []  # base_behavior=="external" の Red エージェントだけ登録

# _set_agents() 変更点
- Red agent 生成時に base_behavior=="external" なら red_external_agents にも追加
- エージェント命名ループを blue → red_external の順で連番付与
- 機体スケール: 4.0 → 0.4（visual_scaleVector）
- 初期オフセット係数: 6 → 0.5（NM スケール変更に合わせた調整）

# 変更された関数
_get_obs_from_agents()    red_external_agents の obs も返すよう拡張
_get_reward_from_agents() 同上
_get_done_from_agents()   同上
_set_agent_actions()      先頭 n_blue 件を agents[]、残りを red_external_agents[] へ振り分け

# 新規関数
_draw_combat_area()       combat_area の 3D ワイヤーフレームを描画
```

#### 2-1-3. `Godot_Air_Combat/addons/godot_rl_agents/B_ACE_sync.gd`

```gdscript
# _send_env_info() 変更点
- all_agents = agents + red_external_agents でラベルと obs_space を収集
- env_info メッセージに "n_blue_agents" フィールドを追加
- "n_agents" が blue + red_external の合計に

# _physics_process() reset ブロック変更点
- obs を obs_dict（agent_name をキーとする辞書）に変換して送信
  （旧: raw obs リストをそのまま送信）

# _physics_process() step ブロック変更点
- _all_agents_step = agents + red_external_agents でループ

# _wait_for_configuration() 変更点
- combat_area は update_dict() をバイパスして直接セット
  （デフォルト Sim Config に存在しないキーのため）
```

#### 2-1-4. `b_ace_py/godot_env.py`

```python
# _get_env_info() に追加
self.n_blue_agents = json_dict.get("n_blue_agents", self.num_envs)
# カスタムバイナリ: Godot が送ってくる n_blue_agents を保存
# 旧バイナリ: フォールバックとして num_envs（全エージェント数）を使用
```

#### 2-1-5. `b_ace_py/B_ACE_GodotPettingZooWrapper.py`

```python
# __init__() 変更点
- _num_agents → _num_blue_agents にリネーム（blue のみの数）
- n_blue = self.n_blue_agents（Godot から取得）
- n_total = self.num_envs（blue + red external）
- possible_agents を n_total 件に変更
- obs_map を blue（ID:101+）と red_external（ID:201+）で分けて構築

# reset() 変更点
- カスタムバイナリ: dict フォーマット（{"agent_0": {"obs":...}}）に対応
- 旧バイナリ: list フォーマット（[{"obs":...}, ...]）にも対応（型判定）

# step() 変更点
- reward 集計は blue エージェント（i < n_blue）のみ対象に変更
  （Red チームのリワードはRL訓練シグナルに含めない）
```

#### 2-1-6. `Examples/MAPPO/vec_env.py`

```python
# PettingZooVecEnv 変更点
- _NM_TO_NORM 定数を追加（NM → 正規化 obs 座標の変換係数）
- _append_boundary_obs(): NM 境界値を正規化してから距離計算するよう修正

# RedTeamVecEnv クラス（新規）
- PettingZooVecEnv のラッパー
- RL アルゴリズム（PPO）からは n_agents = n_blue のみ見える
- reset()/step() で red の obs をキャッシュし、RedTeamPolicy で行動を生成
- 生成した red 行動を blue 行動と結合して inner の step() に渡す
```

#### 2-1-7. `Examples/MAPPO/ppo.py`

```python
# train() 引数追加
red_team_policy: "RedTeamPolicy | None" = None

# 環境構築部分
- red_team_policy が None でない場合: RedTeamVecEnv を使用
- None の場合: 従来通り PettingZooVecEnv を使用
```

#### 2-1-8. `Examples/MAPPO/mappo_b_ace.py`

```python
# 追加
- _resolve_spec_path(): 相対パスを絶対パスに変換（Godot が開くため）
- build_b_ace_config() に combat_area, share_states, share_tracks を追加
- make_reward_fn() の境界チェックを NM → 正規化座標変換に修正
- main ブロックで RedTeamPolicy を構築し train() に渡す
```

---

### 2-2. スペックファイルインターフェース

オリジナルでは機体・ミサイル性能は Godot 内のハードコード値またはデフォルトスペックのみ使用可能だった。

#### 変更点

| ファイル | 変更内容 |
|---|---|
| `Godot_Air_Combat/assets/Default_Sim_Config.json` | `blue_agents`, `red_agents` に `fighter_spec`, `missile_spec` フィールドを追加 |
| `Examples/MAPPO/fighter_spec.json` | 新規作成。`max_g=24.6`（最小旋回半径 0.25 NM @ 650 kts / 25,000 ft）|
| `Examples/MAPPO/missile_spec.json` | 新規作成。デフォルト値と同一内容（カスタマイズ用のコピー） |
| `config.yaml` | `fighter_spec`/`missile_spec` を `res://` パスからローカル JSON ファイルへ変更 |
| `mappo_b_ace.py` | `_resolve_spec_path()` で絶対パス変換。`build_b_ace_config()` に spec パスを渡す |

**最小旋回半径の計算:**
```
r_min = v² / (max_g × GRAVITY_GDM)
v = 650 kts = 3.344 GDM/s, GRAVITY_GDM = 0.0981 GDM/s²
→ max_g = 3.344² / (4.63 × 0.0981) ≈ 24.6  (4.63 GDM = 0.25 NM)
```

---

### 2-3. Combat Area の座標系変更

オリジナルでは `combat_area` の境界値が「正規化 obs 座標」（`own_x_pos` と同スケール）で指定されていた。  
本ブランチでは `config.yaml` の直感性のため **NM（海里）** 指定に変更。

| | オリジナル | 本ブランチ |
|---|---|---|
| 境界指定単位 | 正規化 obs 座標（`own_x_pos` 同スケール） | NM（海里） |
| 例 | `x_min: -0.20` | `x_min: -1` (NM) |
| 変換 | なし | `_NM_TO_NORM = 1852/100/3000 ≈ 0.00617` |

**影響ファイル:**
- `config.yaml`, `eval_config.yaml`: 境界値の数値変更
- `vec_env.py`: `_append_boundary_obs()` に NM → 正規化変換を追加
- `mappo_b_ace.py`: `reward_fn()` の境界チェックに同変換を追加
- `SimManager.gd`: `_draw_combat_area()` で Godot 内に 3D 可視化ボックスを描画

---

### 2-4. シナリオ設定変更

#### `config.yaml`

| 項目 | オリジナル | 本ブランチ |
|---|---|---|
| `env_path` | `B_ACE_v0.1.exe` | `B_ACE_v0.1_custom.exe` |
| `max_cycles` | 36000 | 1200 |
| `blue.init_position.z` | 30.0 NM | 0.8 NM |
| `red.init_position.z` | -30.0 NM | -0.8 NM |
| `red.base_behavior` | `"baseline1"` | `"external"` |
| `num_envs` | 28 | 30 |
| `total_timesteps` | 50,000,000 | 100,000,000 |
| `out_penalty` | -1.0 | -0.0002 |

初期配置を 30 NM → 0.8 NM に縮小したのは、combat_area が ±1 NM の狭域シナリオに合わせるため。

---

### 2-5. 評価スクリプト改善

#### `evaluate.py`

| 変更 | 内容 |
|---|---|
| VecEnv の選択 | `red.base_behavior=="external"` 時は `RedTeamVecEnv` を使用（従来は常に `PettingZooVecEnv`） |
| `_resolve_spec_path()` 追加 | spec ファイルパスを絶対パスに変換 |
| `combat_area` を `EnvConfig` に追加 | Godot にも combat_area を送信 |
| `share_states/share_tracks` を追加 | red config に含める |
| `reset()` 互換対応 | list / dict 両フォーマットを受け付ける（旧バイナリ互換） |
| エラーメッセージ修正 | `config.yaml` → `eval_config.yaml` |
| ファイルオープン | `encoding="utf-8"` 明示 |

#### `eval_config.yaml`

学習設定（`config.yaml`）と完全一致するよう全パラメータを修正。  
以前は以下の不一致があった：

| 項目 | 旧 eval_config | 修正後 |
|---|---|---|
| `max_cycles` | 36000 | 1200（学習と一致） |
| `blue.init_position.z` | 30.0 NM | 0.8 NM |
| `red.base_behavior` | `"baseline1"` | `"external"` |
| `red.init_position.z` | -30.0 NM | -0.8 NM |
| `combat_area` | ±32–40 NM | ±1 NM（学習と一致） |
| `fighter_spec/missile_spec` | なし | `fighter_spec.json` |

---

### 2-6. Godot 描画調整

| ファイル | 変更 | 値 |
|---|---|---|
| `Fighter.gd` | `trail_thickness` | 8.0 → 0.8 |
| `SimManager.gd` | `visual_scaleVector` | 4.0 → 0.4 |
| `SimManager.gd` | 初期オフセット係数 | 6 → 0.5 |
| `ViewPort.gd` | `cameraGlobal.fov`（デフォルト）| 設定なし → 3.0 |
| `ViewPort.gd` | `cameraGlobal.fov`（リセット時）| 設定なし → 30.0 |

スケールを 1/10 に縮小したのは、シナリオの初期配置が 30 NM → 0.8 NM に縮小されたため、  
戦域全体が視野内に収まるよう調整。

---

### 2-7. 依存パッケージ追加

```diff
# requirements.txt
+ torch
+ torchvision
+ tensorboard
```

オリジナルには MAPPO の学習ライブラリ（PyTorch）が含まれていなかった。

---

## 3. アーキテクチャ変更のまとめ図

```
【オリジナル】
  Python (ppo.py)
    └─ PettingZooVecEnv
         └─ B_ACE_GodotPettingZooWrapper  ←→  Godot
              Blue agents: external (RL)
              Red agents:  baseline1 FSM (Godot 内完結)
              n_agents = n_blue

【本ブランチ】
  Python (ppo.py)
    └─ RedTeamVecEnv          ← 新規
         ├─ PettingZooVecEnv
         │    └─ B_ACE_GodotPettingZooWrapper  ←→  Godot (custom binary)
         │         Blue agents: external (RL)
         │         Red agents:  external (Python 制御)  ← 新規
         │         n_agents (Godot) = n_blue + n_red
         └─ RedTeamPolicy     ← 新規
              ├─ チェイスロジック（alive track の aspect_angle）
              ├─ 確率的ミサイル発射
              └─ combat_area 境界回避
         n_agents (RL が見る) = n_blue のみ  ← Red は内部処理
```
