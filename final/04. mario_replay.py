# 04. mario_replay.py
import retro
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
import numpy as np
import sys
import random
import os

relu = lambda X: np.maximum(0, X)
sigmoid = lambda X: 1.0 / (1.0 + np.exp(-X))


class Chromosome:
    def __init__(self):
        self.w1 = np.random.uniform(low=-1, high=1, size=(80, 9))
        self.b1 = np.random.uniform(low=-1, high=1, size=(9,))

        self.w2 = np.random.uniform(low=-1, high=1, size=(9, 6))
        self.b2 = np.random.uniform(low=-1, high=1, size=(6,))

        self.l1 = None

        self.distance = 0
        self.max_distance = 0
        self.frames = 0
        self.stop_frames = 0
        self.win = 0

    def predict(self, data):
        self.l1 = relu(np.matmul(data, self.w1) + self.b1)
        output = sigmoid(np.matmul(self.l1, self.w2) + self.b2)
        result = (output > 0.5).astype(np.int)
        return result

    def fitness(self):
        return int(max(self.distance ** 1.8 - self.frames ** 1.5 + min(max(self.distance - 50, 0), 1) * 2500 + self.win * 1000000, 1))

    # 개체 통계값 초기화
    def clear(self):
        self.distance = 0
        self.max_distance = 0
        self.frames = 0
        self.stop_frames = 0
        self.win = 0


class Mario(QWidget):
    def __init__(self):
        super().__init__()
        self.env = retro.make(game='SuperMarioBros-Nes', state=f'Level1-1')
        screen = self.env.reset()

        self.screen_width = screen.shape[0] * 2
        self.screen_height = screen.shape[1] * 2

        self.screen_tiles_margin_x = 60
        self.screen_tiles_margin_y = 10
        self.neural_network_l1_margin_x = 20
        self.neural_network_w2_margin_x = 10
        self.neural_network_predict_margin_x = 70

        self.setFixedSize(self.screen_width + 400, self.screen_height)

        self.screen_label = QLabel(self)
        self.screen_label.setGeometry(0, 0, self.screen_width, self.screen_height)

        self.info_label = QLabel(self)
        self.info_label.setGeometry(self.screen_width + 320, self.screen_height - 70, 70, 70)
        self.info_label.setText('?????세대\n?번 마리오\n???????')

        self.generation = 12
        self.chromosome_index = 0

        self.current_chromosome = Chromosome()
        self.current_chromosome.w1 = np.load(f'../data/{self.generation}/{self.chromosome_index}/w1.npy')
        self.current_chromosome.b1 = np.load(f'../data/{self.generation}/{self.chromosome_index}/b1.npy')
        self.current_chromosome.w2 = np.load(f'../data/{self.generation}/{self.chromosome_index}/w2.npy')
        self.current_chromosome.b2 = np.load(f'../data/{self.generation}/{self.chromosome_index}/b2.npy')

        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self.update_game)
        self.game_timer.start(1000 // 60)

        self.show()

    def update_screen(self):
        screen = self.env.get_screen()
        qimage = QImage(screen, screen.shape[1], screen.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap(qimage)
        pixmap = pixmap.scaled(self.screen_width, self.screen_height, Qt.IgnoreAspectRatio)
        self.screen_label.setPixmap(pixmap)

    def update_game(self):
        self.update_screen()
        self.update()
        self.info_label.setText(f'{self.generation}세대\n{self.chromosome_index}번 마리오\n{self.current_chromosome.fitness()}')

    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)

        painter.setPen(QPen(Qt.black))

        ram = self.env.get_ram()

        full_screen_tiles = ram[0x0500:0x069F+1]
        full_screen_tile_count = full_screen_tiles.shape[0]

        full_screen_page1_tiles = full_screen_tiles[:full_screen_tile_count // 2].reshape((-1, 16))
        full_screen_page2_tiles = full_screen_tiles[full_screen_tile_count // 2:].reshape((-1, 16))

        full_screen_tiles = np.concatenate((full_screen_page1_tiles, full_screen_page2_tiles), axis=1).astype(np.int)

        enemy_drawn = ram[0x000F:0x0014]
        enemy_horizontal_position_in_level = ram[0x006E:0x0072+1]
        enemy_x_position_on_screen = ram[0x0087:0x008B+1]
        enemy_y_position_on_screen = ram[0x00CF:0x00D3+1]

        for i in range(5):
            if enemy_drawn[i] == 1:
                ex = (((enemy_horizontal_position_in_level[i] * 256) + enemy_x_position_on_screen[i]) % 512 + 8) // 16
                ey = (enemy_y_position_on_screen[i] - 8) // 16 - 1
                if 0 <= ex < full_screen_tiles.shape[1] and 0 <= ey < full_screen_tiles.shape[0]:
                    full_screen_tiles[ey][ex] = -1

        current_screen_in_level = ram[0x071A]
        screen_x_position_in_level = ram[0x071C]
        screen_x_position_offset = (256 * current_screen_in_level + screen_x_position_in_level) % 512
        sx = screen_x_position_offset // 16

        screen_tiles = np.concatenate((full_screen_tiles, full_screen_tiles), axis=1)[:, sx:sx+16]

        for i in range(screen_tiles.shape[0]):
            for j in range(screen_tiles.shape[1]):
                if screen_tiles[i][j] > 0:
                    screen_tiles[i][j] = 1
                if screen_tiles[i][j] == -1:
                    screen_tiles[i][j] = 2
                    painter.setBrush(QBrush(Qt.red))
                else:
                    painter.setBrush(QBrush(QColor.fromHslF(125 / 239, 0 if screen_tiles[i][j] == 0 else 1, 120 / 240)))
                painter.drawRect(self.screen_width + self.screen_tiles_margin_x + 16 * j, self.screen_tiles_margin_y + 16 * i, 16, 16)

        player_x_position_current_screen_offset = ram[0x03AD]
        player_y_position_current_screen_offset = ram[0x03B8]
        px = (player_x_position_current_screen_offset + 8) // 16
        py = (player_y_position_current_screen_offset + 8) // 16 - 1
        painter.setBrush(QBrush(Qt.blue))
        painter.drawRect(self.screen_width + self.screen_tiles_margin_x + 16 * px, self.screen_tiles_margin_y + 16 * py, 16, 16)

        painter.setPen(QPen(Qt.magenta, 2, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        ix = px
        iy = 2
        painter.drawRect(self.screen_width + self.screen_tiles_margin_x + 16 * ix, self.screen_tiles_margin_y + iy * 16, 16 * 8, 16 * 10)

        input_data = screen_tiles[iy:iy+10, ix:ix+8]

        if 2 <= py <= 11:
            input_data[py - 2][0] = 2

        input_data = input_data.flatten()

        self.current_chromosome.frames += 1
        self.current_chromosome.distance = ram[0x006D] * 256 + ram[0x0086]

        if self.current_chromosome.max_distance < self.current_chromosome.distance:
            self.current_chromosome.max_distance = self.current_chromosome.distance
            self.current_chromosome.stop_frame = 0
        else:
            self.current_chromosome.stop_frame += 1

        if ram[0x001D] == 3 or ram[0x0E] in (0x0B, 0x06) or ram[0xB5] == 2 or self.current_chromosome.stop_frame > 180:
            if ram[0x001D] == 3:
                self.current_chromosome.win = 1

            print(f'적합도: {self.current_chromosome.fitness()}')

            self.current_chromosome.clear()
            self.env.reset()
        else:
            predict = self.current_chromosome.predict(input_data)
            press_buttons = np.array([predict[5], 0, 0, 0, predict[0], predict[1], predict[2], predict[3], predict[4]])
            self.env.step(press_buttons)

            for i in range(self.current_chromosome.w2.shape[0]):
                for j in range(self.current_chromosome.w2.shape[1]):
                    if self.current_chromosome.w2[i][j] > 0:
                        painter.setPen(QPen(Qt.red, 1, Qt.SolidLine))
                    else:
                        painter.setPen(QPen(Qt.blue, 1, Qt.SolidLine))
                    painter.drawLine(self.screen_width + self.neural_network_l1_margin_x + self.neural_network_w2_margin_x + i * 40, 252, self.screen_width + self.neural_network_predict_margin_x + self.neural_network_w2_margin_x + j * 40, 452)

            painter.setPen(QPen(Qt.black, 2, Qt.SolidLine))
            for i in range(self.current_chromosome.l1.shape[0]):
                painter.setBrush(QBrush(QColor.fromHslF(125 / 239, 0 if self.current_chromosome.l1[i] == 0 else 1, 120 / 240)))
                painter.drawEllipse(self.screen_width + self.neural_network_l1_margin_x + i * 40, 240, 12 * 2, 12 * 2)

            for i in range(predict.shape[0]):
                painter.setBrush(QBrush(QColor.fromHslF(0.8, 0 if predict[i] <= 0.5 else 1, 0.8)))
                painter.drawEllipse(self.screen_width + self.neural_network_predict_margin_x + i * 40, 440, 12 * 2, 12 * 2)
                text = ('U', 'D', 'L', 'R', 'A', 'B')[i]
                painter.drawText(self.screen_width + self.neural_network_predict_margin_x + i * 40 - 5, 470, text)

        painter.end()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mario = Mario()
    exit(app.exec_())
