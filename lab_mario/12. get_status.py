# 12. get_status.py
import retro

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()

ram = env.get_ram()

# 0x001D	Player "float" state
# 0x03 - 클리어
player_float_state = ram[0x001D]
print(player_float_state)

if player_float_state == 0x03:
    print('클리어')

# 0x000E	Player's state
# 0x06, 0x0B - 게임 오버
player_state = ram[0x000E]
print(player_state)

if player_state == 0x06 or player_state == 0x0B:
    print('게임 오버 1')

# 0x00B5	Player vertical screen position
# anywhere below viewport is >1
player_vertical_screen_position = ram[0x00B5]
print(player_vertical_screen_position)

if player_vertical_screen_position >= 2:
    print('게임 오버 2')
