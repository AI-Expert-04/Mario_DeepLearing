# 10. get_enemy_position.py
import retro

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()

ram = env.get_ram()

# 0x006E-0x0072	Enemy horizontal position in level
# 자신이 속한 화면 페이지 번호
enemy_horizon_position = ram[0x006E:0x0072+1]
# 0x0087-0x008B	Enemy x position on screen
# 자신이 속한 페이지 속 x 좌표
enemy_screen_position_x = ram[0x0087:0x008B+1]
# 0x00CF-0x00D3	Enemy y pos on screen
enemy_position_y = ram[0x00CF:0x00D3+1]
# 적 x 좌표
enemy_position_x = (enemy_horizon_position * 256 + enemy_screen_position_x) % 512

print(enemy_position_x, enemy_position_y)

# 적 타일 좌표
enemy_tile_position_x = (enemy_position_x + 8) // 16
enemy_tile_position_y = (enemy_position_y - 8) // 16 - 1

print(enemy_tile_position_x, enemy_tile_position_y)
