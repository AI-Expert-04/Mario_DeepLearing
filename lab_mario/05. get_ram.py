# 05. get_ram.py
import retro

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()

ram = env.get_ram()

print(ram.shape)
print(ram)

print(ram[0x0003])
