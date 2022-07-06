# 08. get_player_position.py
import retro

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()

ram = env.get_ram()

# 0x03AD	Player x pos within current screen offset
# 현재 화면 속 플레이어 x 좌표
player_position_x = ram[0x03AD]
# 0x03B8	Player y pos within current screen
# 현재 화면 속 플레이어 y 좌표
player_position_y = ram[0x03B8]

print(player_position_x, player_position_y)

# 타일 좌표로 변환
player_tile_position_x = (player_position_x + 8) // 16
player_tile_position_y = (player_position_y + 8) // 16 - 1

print(player_tile_position_x, player_tile_position_y)
