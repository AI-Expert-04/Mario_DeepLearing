# 03. pyqt_paint_event.py
# PyQt 그리기
import sys
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        # 창 크기 고정
        self.setFixedSize(200, 300)
        # 창 제목 설정
        self.setWindowTitle('MyApp')
        # 창 띄우기
        self.show()

    # 창이 업데이트 될 때마다 실행되는 함수
    def paintEvent(self, event):
        # 그리기 도구
        painter = QPainter()
        # 그리기 시작
        painter.begin(self)

        painter.setPen(QPen(Qt.black, 1.0, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.blue))
        painter.drawRect(0, 0, 50, 50)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(50, 0, 50, 50)
        painter.drawRect(0, 50, 50, 50)
        painter.setBrush(QBrush(Qt.red))
        painter.drawRect(50, 50, 50, 50)

        painter.setPen(QPen(Qt.blue, 2.0, Qt.SolidLine))
        painter.drawLine(65, 170, 65, 270)
        painter.setPen(QPen(Qt.red, 2.0, Qt.SolidLine))
        painter.drawLine(20, 170, 65, 270)
        painter.drawLine(110, 170, 65, 270)

        painter.setPen(QPen(Qt.black, 1.0, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.cyan))
        painter.drawEllipse(0, 150, 40, 40)
        painter.drawEllipse(90, 150, 40, 40)
        painter.setBrush(QBrush(Qt.white))
        painter.drawEllipse(45, 150, 40, 40)
        painter.setBrush(QBrush(Qt.gray))
        painter.drawEllipse(45, 250, 40, 40)

        # 그리기 끝
        painter.end()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    sys.exit(app.exec_())

