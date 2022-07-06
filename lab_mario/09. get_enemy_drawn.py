# 09. get_enemy_drawn.py
import retro

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()

ram = env.get_ram()

# 0x000F-0x0013	Enemy drawn? Max 5 enemies at once.
# 0 - No
# 1 - Yes (not so much drawn as "active" or something)
enemy_drawn = ram[0x000F:0x0013+1]

print(enemy_drawn)
