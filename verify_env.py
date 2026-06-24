"""
Run this script after setting up the environment to verify everything works.
Usage: python verify_env.py
"""
import sys


def check(label, fn):
    try:
        result = fn()
        print(f"  [OK]  {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


print("=" * 55)
print(" RL Environment Verification")
print("=" * 55)

failures = 0

# ---------- Python ----------
print("\n[Python]")
check("version", lambda: sys.version.split()[0])

# ---------- PyTorch ----------
print("\n[PyTorch]")
import torch
check("version", lambda: torch.__version__)
check("CUDA available", lambda: torch.cuda.is_available())
if torch.cuda.is_available():
    check("GPU name", lambda: torch.cuda.get_device_name(0))
    check("VRAM (GB)", lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")

# ---------- Gymnasium envs ----------
print("\n[Gymnasium Environments]")
import gymnasium as gym

check("CartPole-v1 (classic_control)", lambda: (
    lambda e: (e.reset(), e.close(), "obs shape (4,)")[2]
)(gym.make("CartPole-v1")))

check("LunarLander-v3 (box2d)", lambda: (
    lambda e: (e.reset(), e.close(), "obs shape (8,)")[2]
)(gym.make("LunarLander-v3")))

def _test_atari():
    import ale_py
    gym.register_envs(ale_py)   # required in gymnasium >= 1.0
    env = gym.make("ALE/Breakout-v5")
    obs, _ = env.reset()
    env.close()
    return f"obs shape {obs.shape}"

ok = check("ALE/Breakout-v5 (atari)", _test_atari)
if not ok:
    failures += 1

# ---------- stable-baselines3 ----------
print("\n[stable-baselines3]")
try:
    import stable_baselines3 as sb3
    check("version", lambda: sb3.__version__)
    # quick smoke test: train PPO on CartPole for 1000 steps
    from stable_baselines3 import PPO
    def _sb3_test():
        model = PPO("MlpPolicy", "CartPole-v1", verbose=0)
        model.learn(total_timesteps=1000)
        return "PPO 1000 steps OK"
    check("PPO smoke test", _sb3_test)
except ImportError as e:
    print(f"  [SKIP] stable-baselines3 not installed: {e}")

# ---------- Summary ----------
print("\n" + "=" * 55)
if failures == 0:
    print(" All checks passed! Ready to train.")
else:
    print(f" {failures} check(s) failed. See messages above.")
print("=" * 55)
