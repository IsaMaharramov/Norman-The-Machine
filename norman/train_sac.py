from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from norman.agents.sac import ReplayBuffer, SACAgent, SACConfig
from norman.envs.load_follow import LoadFollowEnv
from norman.envs.scenarios import BENCHMARK_SCENARIOS
from norman.evaluate import run_episode


def evaluate_agent(agent: SACAgent, seed: int) -> dict:
    rmse, cost, trips = [], 0, 0
    for scenario in BENCHMARK_SCENARIOS:
        env = LoadFollowEnv(scenario=scenario, cycle="MOC")
        _, m = run_episode(env, agent, seed)
        rmse.append(m["rmse_pct"])
        cost += m["cost_steps"]
        trips += int(m["trip"])
    return {"eval_rmse_pct": float(np.mean(rmse)), "eval_cost_steps": cost, "eval_trips": trips}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train the Norman SAC baseline")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--scenario", default="random")
    ap.add_argument("--cycle", default="random")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--warmup-steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device")
    ap.add_argument("--out", default="runs/sac")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    env = LoadFollowEnv(scenario=args.scenario, cycle=args.cycle, hours=args.hours, randomize=True,
                        observation_noise=0.002)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    cfg = SACConfig(batch_size=args.batch_size, warmup_steps=args.warmup_steps)
    agent = SACAgent(obs_dim, act_dim, cfg, args.device)
    buffer = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    log_path = out / "train_log.csv"
    fields = ["episode", "steps", "return", "cost", "trip", "rmse_pct", "alpha", "critic_loss",
              "eval_rmse_pct", "eval_cost_steps", "eval_trips", "wall_s"]
    best = float("inf")
    total_steps = 0
    start = time.perf_counter()
    with log_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done, ep_return, ep_cost, errors, stats = False, 0.0, 0.0, [], {}
            info = {}
            while not done:
                if total_steps < cfg.warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.act(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                buffer.add(obs, action, reward, next_obs, float(terminated))
                obs = next_obs
                ep_return += reward
                ep_cost += info["cost"]
                errors.append(info["error"])
                total_steps += 1
                if total_steps >= cfg.warmup_steps:
                    for _ in range(cfg.updates_per_step):
                        stats = agent.update(buffer, rng)

            row = {
                "episode": ep, "steps": total_steps, "return": round(ep_return, 3), "cost": ep_cost,
                "trip": info.get("trip", False), "rmse_pct": round(float(np.sqrt(np.mean(np.square(errors))) * 100), 3),
                "alpha": round(stats.get("alpha", agent.alpha), 5),
                "critic_loss": round(stats.get("critic_loss", 0.0), 5),
                "wall_s": round(time.perf_counter() - start, 1),
            }
            if (ep + 1) % args.eval_every == 0:
                row.update(evaluate_agent(agent, seed=0))
                agent.save(str(out / "last.pt"), extra={"episode": ep})
                score = row["eval_rmse_pct"] + 0.1 * row["eval_cost_steps"] + 10.0 * row["eval_trips"]
                if score < best:
                    best = score
                    agent.save(str(out / "best.pt"), extra={"episode": ep, **row})
            writer.writerow(row)
            fh.flush()
            print(" | ".join(f"{k}={v}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
