# 02. create_env.py
# 게임 환경 생성
import retro

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()

print(env)
