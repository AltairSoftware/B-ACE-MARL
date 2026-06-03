# 強化学習 データフロー解説

> 対象コード: `Examples/MAPPO/` ディレクトリ  
> アルゴリズム: MAPPO（Multi-Agent Proximal Policy Optimization）

---

## 概要：全体像

```
mappo_b_ace.py
  │  設定読み込み・RedTeamPolicy 構築
  ▼
ppo.py: train()
  │  環境構築・ネットワーク初期化
  ├─ 初期化フェーズ ─────────────────────────────┐
  │                                              │
  │  RedTeamVecEnv                               │
  │    └─ PettingZooVecEnv (n_envs 並列)         │
  │         └─ B_ACE_GodotPettingZooWrapper      │
  │              └─ Godot プロセス (TCP)         │
  └─ 学習ループ ────────────────────────────────┘
       ┌──────────────────────────────┐
       │  venv.reset()               │  reset
       │  for step in range(T):      │
       │    actor.get_action(obs)    │  推論
       │    venv.step(actions)       │  環境ステップ
       │    rollout buffer に記録    │
       │  compute_gae()              │  GAE 計算
       │  ppo_update()               │  ネットワーク更新
       └──────────────────────────────┘
```

---

## Part 1: 初期化フェーズ

### ステップ 1-1: エントリポイント（`mappo_b_ace.py`）

```python
# main ブロック
cfg = yaml.safe_load("config.yaml")          # 設定読み込み
b_ace_config = build_b_ace_config(cfg)       # Godot 向け設定辞書を構築
red_team_policy = RedTeamPolicy(obs_maps={}, # Red team ポリシー仮構築
                                shot_threshold=0.85,
                                combat_area=cfg["combat_area"])
train(cfg, b_ace_config,
      reward_fn=make_reward_fn(cfg),
      red_team_policy=red_team_policy)
```

**`build_b_ace_config()`が生成するデータ構造:**

```python
{
  "EnvConfig": {
    "env_path": "../../bin/B_ACE_v0.1_custom.exe",
    "max_cycles": 1200,
    "combat_area": {"x_min": -1, "x_max": 1, "z_min": -1, "z_max": 1, ...},
    "RewardsConfig": { "hit_enemy_factor": 0.3, ... },
    ...
  },
  "AgentsConfig": {
    "blue_agents": {
      "num_agents": 2, "base_behavior": "external",
      "fighter_spec": "/abs/path/to/fighter_spec.json",  # _resolve_spec_path()
      ...
    },
    "red_agents": {
      "num_agents": 2, "base_behavior": "external",
      "share_states": 1, "share_tracks": 1,
      "fighter_spec": "/abs/path/to/fighter_spec.json",
      ...
    },
  }
}
```

---

### ステップ 1-2: `ppo.py: train()` — 環境構築

```python
env_fns = [lambda idx=i: make_env(b_ace_config, idx) for i in range(n_envs)]
venv = RedTeamVecEnv(env_fns, red_team_policy, combat_area=combat_area)
```

**`RedTeamVecEnv.__init__()`の処理:**

```
RedTeamVecEnv.__init__()
  │
  ├─ PettingZooVecEnv(env_fns, combat_area).__init__()
  │    │
  │    ├─ for each env_fn:
  │    │    make_env(b_ace_config, idx)
  │    │      └─ B_ACE_GodotPettingZooWrapper(**cfg)
  │    │           │
  │    │           ├─ _launch_env()          Godot プロセス起動
  │    │           │    └─ subprocess: B_ACE_v0.1_custom.exe --port=12500+idx
  │    │           │
  │    │           ├─ _start_server()        TCP サーバー待機（Python側）
  │    │           │
  │    │           ├─ _handshake()           バージョン確認
  │    │           │    Python →(JSON)→ Godot: {"type":"handshake","major":"0",...}
  │    │           │    Godot  →(JSON)→ Python: 同形式で応答
  │    │           │
  │    │           ├─ send_sim_config()      設定を Godot へ送信
  │    │           │    Python →(JSON)→ Godot:
  │    │           │      {"type":"config",
  │    │           │       "env_config": EnvConfig,
  │    │           │       "agents_config": AgentsConfig}
  │    │           │    Godot 側: B_ACE_sync._wait_for_configuration()
  │    │           │      → SimManager.initialize() → _set_agents()
  │    │           │        Blue: agents[] に追加
  │    │           │        Red(external): enemies[] + red_external_agents[] に追加
  │    │           │      → _draw_combat_area() で 3D 枠を描画
  │    │           │
  │    │           └─ _get_env_info()        観測/行動空間の情報取得
  │    │                Python →(JSON)→ Godot: {"type":"env_info"}
  │    │                Godot  →(JSON)→ Python:
  │    │                  {
  │    │                    "type": "env_info",
  │    │                    "n_agents": 4,           ← blue(2) + red_ext(2)
  │    │                    "n_blue_agents": 2,      ← カスタムバイナリが追加
  │    │                    "observation_space": {"obs": {"size":[41],"space":"box"}},
  │    │                    "action_space": {"input": {"action_type":"continuous","size":4}},
  │    │                    "observation_labels": {
  │    │                      "101": ["own_x_pos","own_z_pos",...,"track_detected_201",...],
  │    │                      "102": [...],
  │    │                      "201": [...,"track_detected_101",...],
  │    │                      "202": [...]
  │    │                    }
  │    │                  }
  │    │                ↓
  │    │                self.num_envs = 4  (n_agents)
  │    │                self.n_blue_agents = 2
  │    │
  │    │           Wrapper の obs_map 構築:
  │    │             "agent_0" → {label: idx} for Godot ID 101 (blue)
  │    │             "agent_1" → {label: idx} for Godot ID 102 (blue)
  │    │             "agent_2" → {label: idx} for Godot ID 201 (red_ext)
  │    │             "agent_3" → {label: idx} for Godot ID 202 (red_ext)
  │    │
  │    ├─ ref = envs[0]
  │    ├─ n_agents = 4, n_blue = 2 (from ref.n_blue)
  │    ├─ obs_dim = 41 + 4 = 45  (base + combat_area 4値)
  │    └─ action_dim = 4
  │
  ├─ n_blue = inner.envs[0].n_blue  →  2
  ├─ _n_red = 4 - 2 = 2
  ├─ _red_agent_names = ["agent_2", "agent_3"]
  │
  └─ obs_maps を RedTeamPolicy に注入:
       red_team_policy.obs_maps = {
         "agent_2": {"own_x_pos":0, ..., "track_detected_101":..., ...},
         "agent_3": {"own_x_pos":0, ..., "track_detected_101":..., ...},
       }
```

---

### ステップ 1-3: ネットワーク初期化

```python
actor  = MAPPOActor(obs_dim=45, action_dim=4, hidden=[256, 256])
critic = MAPPOCritic(global_obs_dim=45*2=90, hidden=[256, 256])
```

- **Actor**: 各 Blue エージェントの局所観測 (45次元) → 行動 (4次元) の連続確率分布（Diagonal Gaussian）
- **Critic**: 全 Blue エージェントの観測を結合した大域観測 (90次元) → 状態価値 (スカラー)

---

## Part 2: 学習実行フェーズ

### ステップ 2-1: 環境リセット

```
train() → venv.reset()
  │
  └─ RedTeamVecEnv.reset()
       │
       └─ PettingZooVecEnv.reset()
            │
            ├─ ThreadPoolExecutor で並列実行
            │    └─ _reset_one(env):
            │         env.reset()  ← B_ACE_GodotPettingZooWrapper.reset()
            │           │
            │           └─ GodotEnv.reset()
            │                Python →(JSON)→ Godot: {"type":"reset"}
            │                Godot:
            │                  _reset_simulation() → 全 Fighter.reset()
            │                  just_reset = true
            │                  次の physics tick で obs を収集し送信:
            │                    obs = _get_obs_from_agents()
            │                      ※ agents[] + red_external_agents[] 全員分
            │                    obs_dict = {
            │                      "agent_0": {"obs": [0.0, 0.005, ...]},  ← blue1
            │                      "agent_1": {"obs": [...]},               ← blue2
            │                      "agent_2": {"obs": [...]},               ← red1
            │                      "agent_3": {"obs": [...]},               ← red2
            │                    }
            │                Python ←(JSON)← Godot: {"type":"reset","obs":obs_dict}
            │
            │           Wrapper.reset() がフォーマット判定:
            │             dict → agent_name: {"obs":..., "mask":[True,True,True,True]}
            │             list → enumerate でインデックス付け（旧バイナリ互換）
            │           return self.observations  ← 全エージェント分
            │
            │         _extract(obs_dict)
            │           np.stack([obs["obs"] for a in all_agents]) → (4, 41)
            │         _append_boundary_obs((4, 41)) → (4, 45)
            │           NM境界 → 正規化座標 (× _NM_TO_NORM=0.00617)
            │           [dist_to_x_min, dist_to_x_max, dist_to_z_min, dist_to_z_max]
            │         return (4, 45)
            │
            └─ 全 env の obs を stack → (n_envs, 4, 45)
       │
       ├─ _last_red_obs = all_obs[:, 2:, :] → (n_envs, 2, 45)  ← Red obs をキャッシュ
       └─ return all_obs[:, :2, :] → (n_envs, 2, 45)           ← Blue obs のみ返す

obs = (30, 2, 45)   shape: (n_envs, n_blue, obs_dim)
```

---

### ステップ 2-2: ロールアウト収集ループ

`for step in range(T=1024):` の1ステップ：

#### 2-2-1: 推論

```python
obs_flat = obs.reshape(n_envs * n_blue, obs_dim)  # (60, 45)
obs_t = torch.as_tensor(obs_flat)                  # (60, 45)

actions_t, logprobs_t, _ = actor.get_action(obs_t)
# Actor 内部:
#   mean = network(obs_t)      → (60, 4)  tanh で [-1,1] にクリップ
#   dist = Normal(mean, std)
#   actions = dist.rsample()   → (60, 4)  再パラメトリゼーション
#   logprobs = dist.log_prob() → (60, 4)  次元ごとの log p を合計

values_t = critic(global_obs_t)   # global_obs = obs 全エージェントを結合
# global_obs shape: (n_envs, n_blue * obs_dim) = (30, 90)
# critic: Linear(90→256) → ReLU → ... → Linear(256→1) → (30, 1)

actions_np = actions_t.numpy().reshape(n_envs, n_blue, action_dim)
# (30, 2, 4)  [heading, altitude, desired_g, fire] × n_blue
```

#### 2-2-2: 環境ステップ

```
venv.step(actions_np)  ← (30, 2, 4) blue のみ
  │
  └─ RedTeamVecEnv.step(blue_actions)
       │
       ├─ _compute_red_actions()
       │    for e in range(30):
       │      RedTeamPolicy.act_batch(
       │        ["agent_2","agent_3"],
       │        _last_red_obs[e]          ← (2, 45)
       │      )
       │      →  act("agent_2", obs[0]):
       │           _best_target()
       │             labels["track_dist_101"]: dist=-1? → 死亡判定
       │             labels["track_dist_102"]: dist≥0  → alive
       │             aspect = obs[labels["track_aspect_angle_102"]]  ← [-1,1]
       │             offensive = obs[labels["track_offensive_factor_102"]] + 1.0
       │             is_detected = obs[labels["track_detected_102"]] > 0.5
       │           _boundary_heading()
       │             x = obs[labels["own_x_pos"]]
       │             z = obs[labels["own_z_pos"]]
       │             hdg = obs[labels["own_current_hdg"]] * 180°
       │             margin = 0.2 NM * 0.00617 = 0.001234
       │             min_dist = min(各壁までの距離)
       │             strength = clip(1 - min_dist/margin, 0, 1)
       │             desired_hdg = atan2(x_ctr-x, -(z_ctr-z)) [度]
       │             boundary_cmd = (desired_hdg - hdg) / 180 → [-1,1]
       │           return [heading_cmd, alt_cmd, g_cmd, fire_cmd]
       │    → red_actions: (30, 2, 4)
       │
       ├─ all_actions = concat([blue(30,2,4), red(30,2,4)], axis=1)
       │    → (30, 4, 4)
       │
       └─ PettingZooVecEnv.step(all_actions)
            │
            ├─ ThreadPoolExecutor で n_envs=30 並列
            │    └─ _step_one(env, actions_dict):
            │         actions_dict = {
            │           "agent_0": all_actions[e,0].tolist(),  ← blue1 行動
            │           "agent_1": all_actions[e,1].tolist(),  ← blue2 行動
            │           "agent_2": all_actions[e,2].tolist(),  ← red1 行動
            │           "agent_3": all_actions[e,3].tolist(),  ← red2 行動
            │         }
            │         env.step(actions_dict)
            │           ↓ B_ACE_GodotPettingZooWrapper.step()
            │
            │         godot_actions = np.array([
            │           [[h0,a0,g0,f0]],   ← agent_0 action (shape: 1,4)
            │           [[h1,a1,g1,f1]],
            │           [[h2,a2,g2,f2]],
            │           [[h3,a3,g3,f3]],
            │         ])  shape: (4, 1, 4)
            │
            │         step_send(godot_actions, order_ij=True)
            │           from_numpy(action, order_ij=True):
            │             for agent_idx in range(4):  ← num_envs=4 (total agents)
            │               env_action["input"] = action[agent_idx][0].tolist()
            │             → [{"input":[h0,a0,g0,f0]}, {"input":[h1,...]},
            │                {"input":[h2,...]}, {"input":[h3,...]}]
            │           Python →(JSON)→ Godot:
            │             {"type":"action", "action":[{...},{...},{...},{...}]}
            │
            │         Godot 側: _set_agent_actions(actions)
            │           n_blue = len(agents) = 2
            │           agents[0].set_action(actions[0]) ← blue1 に行動設定
            │           agents[1].set_action(actions[1]) ← blue2 に行動設定
            │           red_external_agents[0].set_action(actions[2]) ← red1
            │           red_external_agents[1].set_action(actions[3]) ← red2
            │
            │         Godot: action_repeat=20 回の physics ステップを実行
            │           各 Fighter._physics_process(delta):
            │             process_tracks()          ← レーダー更新
            │             process_allied_tracks()   ← 味方データリンク
            │             if behavior != "external":
            │               process_behavior()      ← FSM (external は skip)
            │             if behavior == "external" and shoot_input > 0:
            │               if abs(HPT.aspect_angle) < 30.0:
            │                 launch_missile_at_target(HPT)
            │             turn_g = desiredG * GRAVITY_GDM
            │             turn_speed = turn_g / velocity.length()
            │             transform.basis = rotated(UP, -turn_input * turn_speed * delta)
            │             velocity = -basis.z * current_speed
            │
            │         Godot: obs/reward/done を収集・送信
            │           _get_obs_from_agents()  ← agents[] + red_external_agents[]
            │           _get_reward_from_agents()  ← 同上
            │           _get_done_from_agents()    ← 同上
            │           → {"type":"step", "obs":obs_dict, "reward":..., "done":...}
            │
            │         step_recv()
            │           Python ←(JSON)← Godot: {"type":"step", ...}
            │           return (obs_dict, reward_dict, done_dict, ...)
            │
            │         Wrapper.step() 後処理:
            │           observations = {agent: {"obs":[...], "mask":[...]} for all}
            │           for i, agent in enumerate(possible_agents):
            │             terminations |= dones[agent]
            │             if i < n_blue:
            │               rewards += reward[agent]  ← Blue のみ集計
            │           return observations, rewards(スカラー), terminations, ...
            │
            │         done した場合: env.reset() を呼び obs を再初期化
            │
            ├─ 全 env の結果を集約
            │    obs_list:    [(4,45), ...] → stack → (30, 4, 45)
            │    reward_list: [float, ...] → array → (30,)
            │    done_list:   [float, ...] → array → (30,)
            │
            └─ return (30,4,45), (30,), (30,), infos
       │
       ├─ _last_red_obs = all_obs[:, 2:, :] → 更新
       └─ return blue_obs (30,2,45), rewards(30,), dones(30,), infos
```

#### 2-2-3: reward_fn 適用（combat_area ペナルティ）

```python
if reward_fn is not None:
    rewards = [reward_fn(next_obs[e], float(rewards[e]), bool(dones[e]))
               for e in range(n_envs)]

# make_reward_fn() が返す reward_fn:
def reward_fn(obs, reward, done):
    # obs shape: (n_blue, obs_dim)
    x_pos = obs[:, 0]  # own_x_pos (正規化済み)
    z_pos = obs[:, 1]  # own_z_pos (正規化済み)
    k = 1852/100/3000  # NM → 正規化変換係数
    outside = (
      any(x_pos < -1 * k) or any(x_pos > 1 * k) or
      any(z_pos < -1 * k) or any(z_pos > 1 * k)
    )
    if outside:
        reward += -0.0002  # out_penalty
    return reward
```

#### 2-2-4: ロールアウトバッファへの記録

```python
b_local_obs[step]  = obs          # (n_envs, n_blue, obs_dim)
b_global_obs[step] = get_global_obs(obs)  # (n_envs, n_blue*obs_dim)
b_actions[step]    = actions_np    # (n_envs, n_blue, action_dim)
b_logprobs[step]   = logprobs_np   # (n_envs, n_blue)
b_rewards[step]    = rewards       # (n_envs,)
b_dones[step]      = dones         # (n_envs,)
b_values[step]     = values        # (n_envs,)
```

---

### ステップ 2-3: Generalized Advantage Estimation (GAE)

T=1024 ステップ収集後:

```python
# ブートストラップ価値
with torch.no_grad():
    next_values = critic(next_global_obs).squeeze()  # (n_envs,)

# 各環境ごとに GAE 計算
for e in range(n_envs):
    advantages[:,e], returns[:,e] = compute_gae(
        rewards   = b_rewards[:, e],    # (T,)
        values    = b_values[:, e],     # (T,)
        dones     = b_dones[:, e],      # (T,)
        next_value = next_values[e],    # スカラー
        next_done  = last_done[e],
        gamma=0.99, gae_lambda=0.95
    )

# compute_gae() 内部:
# δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
# A_t = δ_t + γλ * A_{t+1}
# G_t = A_t + V(s_t)  （returns）
```

---

### ステップ 2-4: PPO ミニバッチ更新

```python
# フラット化: N = T * n_envs = 1024 * 30 = 30720
flat_obs     = b_local_obs.reshape(N*n_blue, obs_dim)  # (61440, 45)
flat_actions = b_actions.reshape(N*n_blue, action_dim)  # (61440, 4)
flat_adv     = repeat(advantages, n_blue)              # (61440,)

# 4エポック × 4ミニバッチ = 16 回更新
for epoch in range(4):
    perm = torch.randperm(N * n_blue)
    for start in range(0, N*n_blue, mb_size=7680):
        mb = perm[start:start+mb_size]

        new_logprob, entropy = actor.get_log_prob_entropy(flat_obs[mb], flat_actions[mb])
        new_value = critic(flat_global[mb // n_blue])

        # PPO クリップ損失（Actor）
        ratio = exp(new_logprob - old_logprob[mb])
        pg_loss = max(-adv * ratio, -adv * clip(ratio, 1-0.2, 1+0.2)).mean()

        # Value 損失（Critic）
        vf_loss = max((new_value - returns)², (clip(new_value) - returns)²).mean()

        # エントロピーボーナス
        loss = pg_loss - 0.01 * entropy.mean() + 0.5 * vf_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(params, 0.5)
        optimizer.step()
```

---

## Part 3: 観測・行動の詳細フォーマット

### 観測ベクトル構造（dim=41 + 4=45）

```
インデックス  ラベル                       値の意味
─────────────────────────────────────────────────────────────
 0  own_x_pos                  global_x / 3000.0
 1  own_z_pos                  global_z / 3000.0
 2  own_altitude               global_y / 150.0
 3  own_dist_target            dist_to_target / 3000.0
 4  own_aspect_angle_target    aspect_to_target / 180.0
 5  own_current_hdg            current_hdg_deg / 180.0
 6  own_current_speed          speed / max_speed
 7  own_missiles               missiles / 6.0
 8  own_in_flight_missile      0 or 1
─ 敵トラック #1 (ID=敵エージェントID) ×13 ──────────────────
 9  track_alt_diff             Δy / 150.0
10  track_aspect_angle         aspect_angle / 180.0  ← チェイスに使用
11  track_angle_off            angle_off / 180.0
12  track_dist                 dist / 3000.0  (-1.0=dead sentinel)
13  track_dist2go              dist2go / 3000.0
14  track_own_missile_RMax     RMax / 926.0
15  track_own_missile_Nez      Nez / 926.0
16  track_enemy_missile_RMax   RMax / 926.0
17  track_enemy_missile_Nez    Nez / 926.0
18  track_threat_factor        threat_factor - 1
19  track_offensive_factor     offensive_factor - 1  ← 発射判定に使用
20  track_is_missile_support   0 or 1
21  track_detected             0 or 1  ← 直接 detected かどうか
─ 敵トラック #2 ×13 (22〜34) ─────────────────────────────
─ 味方トラック #1 ×6 (35〜40) ────────────────────────────
─ combat_area 境界距離 ×4 (41〜44) ─────────────────────
41  dist_to_x_min  (x - x_min) / area_w     正:内側, 負:外側
42  dist_to_x_max  (x_max - x) / area_w
43  dist_to_z_min  (z - z_min) / area_d
44  dist_to_z_max  (z_max - z) / area_d
```

### 行動ベクトル（dim=4）

```
インデックス  意味                     変換（Godot 側）
─────────────────────────────────────────────────────────────
 0  heading   [-1, 1]  →  ±180° の相対旋回
                          hdg_input = current_hdg + action[0]*180
                          turn_input = clamp((hdg_diff)/60, -1, 1)
 1  altitude  [-1, 1]  →  [0, 50000] ft
                          level_input = (action[1]*25000+25000) * FT2GDM
 2  desired_g [-1, 1]  →  [1, max_g] G
                          desiredG = (action[2]*(max_g-1)+(max_g+1))/2
 3  fire      ≤0=不発, >0=発射
                          Godot がさらに条件チェック:
                          HPT != null AND abs(HPT.aspect_angle) < 30°
```

---

## Part 4: TCP 通信プロトコル

```
Python                          Godot (B_ACE_sync.gd)
──────────────────────────────────────────────────────
→ {"type":"handshake",...}
                            ← {"type":"handshake",...}
→ {"type":"config", "env_config":{...}, "agents_config":{...}}
→ {"type":"env_info"}
                            ← {"type":"env_info", "n_agents":4,
                               "n_blue_agents":2, "observation_labels":{...}, ...}
                            [ physics ループ開始 ]
→ {"type":"reset"}
                            ← {"type":"reset", "obs":{"agent_0":{...},...}}
→ {"type":"action", "action":[{"input":[...]}, ...]}
                            [ action_repeat=20 回 physics step ]
                            ← {"type":"step", "obs":{...},
                               "reward":{...}, "done":{...}}
→ {"type":"action", ...}      ← 以降繰り返し
...
→ {"type":"reset"}            ← エピソード終了後
...
→ {"type":"close"}
                            [ Godot プロセス終了 ]
```

---

## Part 5: データシェイプ早見表

| 変数 | シェイプ | 意味 |
|---|---|---|
| `obs` (学習中) | (30, 2, 45) | n_envs × n_blue × obs_dim |
| `all_obs` (内部) | (30, 4, 45) | n_envs × (n_blue+n_red) × obs_dim |
| `actions_np` | (30, 2, 4) | n_envs × n_blue × action_dim |
| `red_actions` | (30, 2, 4) | n_envs × n_red × action_dim |
| `all_actions` | (30, 4, 4) | n_envs × n_all × action_dim |
| `godot_actions` | (4, 1, 4) | n_all × 1 × action_dim（1env分） |
| `b_local_obs` | (1024, 30, 2, 45) | T × n_envs × n_blue × obs_dim |
| `b_global_obs` | (1024, 30, 90) | T × n_envs × (n_blue×obs_dim) |
| `flat_obs` | (61440, 45) | (T×n_envs×n_blue) × obs_dim |
| `flat_global` | (30720, 90) | (T×n_envs) × global_obs_dim |
